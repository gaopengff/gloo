/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/allreduce.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"

namespace gloo {

namespace {

using BufferVector = std::vector<std::unique_ptr<transport::UnboundBuffer>>;
using ReductionFunction = AllreduceOptions::Func;
using ReduceRangeFunction = std::function<void(size_t, size_t)>;
using BroadcastRangeFunction = std::function<void(size_t, size_t)>;

// Forward declaration of ring algorithm implementation.
void ring(
    const detail::AllreduceOptionsImpl& opts,
    ReduceRangeFunction reduceInputs,
    BroadcastRangeFunction broadcastOutputs);

// Forward declaration of bcube algorithm implementation.
void bcube(
    const detail::AllreduceOptionsImpl& opts,
    ReduceRangeFunction reduceInputs,
    BroadcastRangeFunction broadcastOutputs);

void shm(const detail::AllreduceOptionsImpl& opts);

// Returns function that computes local reduction over inputs and
// stores it in the output for a given range in those buffers.
// This is done prior to either sending a region to a neighbor, or
// reducing a region received from a neighbor.
ReduceRangeFunction genLocalReduceFunction(
    const BufferVector& in,
    const BufferVector& out,
    size_t elementSize,
    ReductionFunction fn) {
  if (in.size() > 0) {
    if (in.size() == 1) {
      return [&in, &out](size_t offset, size_t length) {
        memcpy(
            static_cast<uint8_t*>(out[0]->ptr) + offset,
            static_cast<const uint8_t*>(in[0]->ptr) + offset,
            length);
      };
    } else {
      return [&in, &out, elementSize, fn](size_t offset, size_t length) {
        fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(in[0]->ptr) + offset,
           static_cast<const uint8_t*>(in[1]->ptr) + offset,
           length / elementSize);
        for (size_t i = 2; i < in.size(); i++) {
          fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
             static_cast<const uint8_t*>(out[0]->ptr) + offset,
             static_cast<const uint8_t*>(in[i]->ptr) + offset,
             length / elementSize);
        }
      };
    }
  } else {
    return [&out, elementSize, fn](size_t offset, size_t length) {
      for (size_t i = 1; i < out.size(); i++) {
        fn(static_cast<uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(out[0]->ptr) + offset,
           static_cast<const uint8_t*>(out[i]->ptr) + offset,
           length / elementSize);
      }
    };
  }
}

// Returns function that performs a local broadcast over outputs for a
// given range in the buffers. This is executed after receiving every
// globally reduced chunk.
BroadcastRangeFunction genLocalBroadcastFunction(const BufferVector& out) {
  return [&out](size_t offset, size_t length) {
    for (size_t i = 1; i < out.size(); i++) {
      memcpy(
          static_cast<uint8_t*>(out[i]->ptr) + offset,
          static_cast<const uint8_t*>(out[0]->ptr) + offset,
          length);
    }
  };
}

void allreduce(const detail::AllreduceOptionsImpl& opts) {
  if (opts.elements == 0) {
    return;
  }

  const auto& context = opts.context;
  const std::vector<std::unique_ptr<transport::UnboundBuffer>>& in = opts.in;
  const std::vector<std::unique_ptr<transport::UnboundBuffer>>& out = opts.out;

  // Sanity checks
  GLOO_ENFORCE_GT(out.size(), 0);
  GLOO_ENFORCE(opts.elementSize > 0);
  GLOO_ENFORCE(opts.reduce != nullptr);

  // Assert the size of all inputs and outputs is identical.
  const size_t totalBytes = opts.elements * opts.elementSize;
  for (size_t i = 0; i < out.size(); i++) {
    GLOO_ENFORCE_EQ(out[i]->size, totalBytes);
  }
  for (size_t i = 0; i < in.size(); i++) {
    GLOO_ENFORCE_EQ(in[i]->size, totalBytes);
  }

  // Initialize local reduction and broadcast functions.
  // Note that these are a no-op if only a single output is specified
  // and is used as both input and output.
  const auto reduceInputs =
      genLocalReduceFunction(in, out, opts.elementSize, opts.reduce);
  const auto broadcastOutputs = genLocalBroadcastFunction(out);

  // Simple circuit if there is only a single process.
  if (context->size == 1) {
    reduceInputs(0, totalBytes);
    broadcastOutputs(0, totalBytes);
    return;
  }

  switch (opts.algorithm) {
    case detail::AllreduceOptionsImpl::UNSPECIFIED:
    case detail::AllreduceOptionsImpl::RING:
      ring(opts, reduceInputs, broadcastOutputs);
      break;
    case detail::AllreduceOptionsImpl::BCUBE:
      bcube(opts, reduceInputs, broadcastOutputs);
      break;
    case detail::AllreduceOptionsImpl::SHM:
      shm(opts);
      break;
    default:
      GLOO_ENFORCE(false, "Algorithm not handled.");
  }
}

void ring(
    const detail::AllreduceOptionsImpl& opts,
    ReduceRangeFunction reduceInputs,
    BroadcastRangeFunction broadcastOutputs) {
  const auto& context = opts.context;
  const std::vector<std::unique_ptr<transport::UnboundBuffer>>& out = opts.out;
  const auto slot = Slot::build(kAllreduceSlotPrefix, opts.tag);
  const size_t totalBytes = opts.elements * opts.elementSize;

  // Note: context->size > 1
  const auto recvRank = (context->size + context->rank + 1) % context->size;
  const auto sendRank = (context->size + context->rank - 1) % context->size;
  GLOO_ENFORCE(
      context->getPair(recvRank),
      "missing connection between rank " + std::to_string(context->rank) +
          " (this process) and rank " + std::to_string(recvRank));
  GLOO_ENFORCE(
      context->getPair(sendRank),
      "missing connection between rank " + std::to_string(context->rank) +
          " (this process) and rank " + std::to_string(sendRank));

  // The ring algorithm works as follows.
  //
  // The given input is split into a number of chunks equal to the
  // number of processes. Once the algorithm has finished, every
  // process hosts one chunk of reduced output, in sequential order
  // (rank 0 has chunk 0, rank 1 has chunk 1, etc.). As the input may
  // not be divisible by the number of processes, the chunk on the
  // final ranks may have partial output or may be empty.
  //
  // As a chunk is passed along the ring and contains the reduction of
  // successively more ranks, we have to alternate between performing
  // I/O for that chunk and computing the reduction between the
  // received chunk and the local chunk. To avoid this alternating
  // pattern, we split up a chunk into multiple segments (>= 2), and
  // ensure we have one segment in flight while computing a reduction
  // on the other. The segment size has an upper bound to minimize
  // memory usage and avoid poor cache behavior. This means we may
  // have many segments per chunk when dealing with very large inputs.
  //
  // The nomenclature here is reflected in the variable naming below
  // (one chunk per rank and many segments per chunk).
  //

  // Ensure that maximum segment size is a multiple of the element size.
  // Otherwise, the segment size can exceed the maximum segment size after
  // rounding it up to the nearest multiple of the element size.
  // For example, if maxSegmentSize = 10, and elementSize = 4,
  // then after rounding up: segmentSize = 12;
  const size_t maxSegmentBytes = opts.elementSize *
      std::max((size_t)1, opts.maxSegmentSize / opts.elementSize);

  // Compute how many segments make up the input buffer.
  //
  // Round up to the nearest multiple of the context size such that
  // there is an equal number of segments per process and execution is
  // symmetric across processes.
  //
  // The minimum is twice the context size, because the algorithm
  // below overlaps sending/receiving a segment with computing the
  // reduction of the another segment.
  //
  const size_t numSegments = roundUp(
      std::max(
          (totalBytes + (maxSegmentBytes - 1)) / maxSegmentBytes,
          (size_t)context->size * 2),
      (size_t)context->size);
  GLOO_ENFORCE_EQ(numSegments % context->size, 0);
  GLOO_ENFORCE_GE(numSegments, context->size * 2);
  const size_t numSegmentsPerRank = numSegments / context->size;
  const size_t segmentBytes =
      roundUp((totalBytes + numSegments - 1) / numSegments, opts.elementSize);

  // Allocate scratch space to hold two chunks
  std::unique_ptr<uint8_t[]> tmpAllocation(new uint8_t[segmentBytes * 2]);
  std::unique_ptr<transport::UnboundBuffer> tmpBuffer =
      context->createUnboundBuffer(tmpAllocation.get(), segmentBytes * 2);
  transport::UnboundBuffer* tmp = tmpBuffer.get();

  // Use dynamic lookup for chunk offset in the temporary buffer.
  // With two operations in flight we need two offsets.
  // They can be indexed using the loop counter.
  std::array<size_t, 2> segmentOffset;
  segmentOffset[0] = 0;
  segmentOffset[1] = segmentBytes;

  // Function computes the offsets and lengths of the segments to be
  // sent and received for a given iteration during reduce/scatter.
  auto computeReduceScatterOffsets = [&](size_t i) {
    struct {
      size_t sendOffset;
      size_t recvOffset;
      ssize_t sendLength;
      ssize_t recvLength;
    } result;

    // Compute segment index to send from (to rank - 1) and segment
    // index to receive into (from rank + 1). Multiply by the number
    // of bytes in a chunk to get to an offset. The offset is allowed
    // to be out of range (>= totalBytes) and this is taken into
    // account when computing the associated length.
    result.sendOffset =
        ((((context->rank + 1) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);
    result.recvOffset =
        ((((context->rank + 2) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);

    // If the segment is entirely in range, the following statement is
    // equal to segmentBytes. If it isn't, it will be less, or even
    // negative. This is why the ssize_t typecasts are needed.
    result.sendLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.sendOffset);
    result.recvLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.recvOffset);

    return result;
  };

  // Ring reduce/scatter.
  //
  // Number of iterations is computed as follows:
  // - Take `numSegments` for the total number of segments,
  // - Subtract `numSegmentsPerRank` because the final segments hold
  //   the partial result and must not be forwarded in this phase.
  // - Add 2 because we pipeline send and receive operations (we issue
  //   send/recv operations on iterations 0 and 1 and wait for them to
  //   complete on iterations 2 and 3).
  //
  for (auto i = 0; i < (numSegments - numSegmentsPerRank + 2); i++) {
    if (i >= 2) {
      // Compute send and receive offsets and lengths two iterations
      // ago. Needed so we know when to wait for an operation and when
      // to ignore (when the offset was out of bounds), and know where
      // to reduce the contents of the temporary buffer.
      auto prev = computeReduceScatterOffsets(i - 2);
      if (prev.recvLength > 0) {
        // Prepare out[0]->ptr to hold the local reduction
        reduceInputs(prev.recvOffset, prev.recvLength);
        // Wait for segment from neighbor.
        tmp->waitRecv(opts.timeout);
        // Reduce segment from neighbor into out->ptr.
        opts.reduce(
            static_cast<uint8_t*>(out[0]->ptr) + prev.recvOffset,
            static_cast<const uint8_t*>(out[0]->ptr) + prev.recvOffset,
            static_cast<const uint8_t*>(tmp->ptr) + segmentOffset[i & 0x1],
            prev.recvLength / opts.elementSize);
      }
      if (prev.sendLength > 0) {
        out[0]->waitSend(opts.timeout);
      }
    }

    // Issue new send and receive operation in all but the final two
    // iterations. At that point we have already sent all data we
    // needed to and only have to wait for the final segments to be
    // reduced into the output.
    if (i < (numSegments - numSegmentsPerRank)) {
      // Compute send and receive offsets and lengths for this iteration.
      auto cur = computeReduceScatterOffsets(i);
      if (cur.recvLength > 0) {
        tmp->recv(recvRank, slot, segmentOffset[i & 0x1], cur.recvLength);
      }
      if (cur.sendLength > 0) {
        // Prepare out[0]->ptr to hold the local reduction for this segment
        if (i < numSegmentsPerRank) {
          reduceInputs(cur.sendOffset, cur.sendLength);
        }
        out[0]->send(sendRank, slot, cur.sendOffset, cur.sendLength);
      }
    }
  }

  // Function computes the offsets and lengths of the segments to be
  // sent and received for a given iteration during allgather.
  auto computeAllgatherOffsets = [&](size_t i) {
    struct {
      size_t sendOffset;
      size_t recvOffset;
      ssize_t sendLength;
      ssize_t recvLength;
    } result;

    result.sendOffset =
        ((((context->rank) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);
    result.recvOffset =
        ((((context->rank + 1) * numSegmentsPerRank) + i) * segmentBytes) %
        (numSegments * segmentBytes);

    // If the segment is entirely in range, the following statement is
    // equal to segmentBytes. If it isn't, it will be less, or even
    // negative. This is why the ssize_t typecasts are needed.
    result.sendLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.sendOffset);
    result.recvLength = std::min(
        (ssize_t)segmentBytes,
        (ssize_t)totalBytes - (ssize_t)result.recvOffset);

    return result;
  };

  // Ring allgather.
  //
  // Beware: totalBytes <= (numSegments * segmentBytes), which is
  // incompatible with the generic allgather algorithm where the
  // contribution is identical across processes.
  //
  // See comment prior to reduce/scatter loop on how the number of
  // iterations for this loop is computed.
  //
  for (auto i = 0; i < (numSegments - numSegmentsPerRank + 2); i++) {
    if (i >= 2) {
      auto prev = computeAllgatherOffsets(i - 2);
      if (prev.recvLength > 0) {
        out[0]->waitRecv(opts.timeout);
        // Broadcast received segments to output buffers.
        broadcastOutputs(prev.recvOffset, prev.recvLength);
      }
      if (prev.sendLength > 0) {
        out[0]->waitSend(opts.timeout);
      }
    }

    // Issue new send and receive operation in all but the final two
    // iterations. At that point we have already sent all data we
    // needed to and only have to wait for the final segments to be
    // sent to the output.
    if (i < (numSegments - numSegmentsPerRank)) {
      auto cur = computeAllgatherOffsets(i);
      if (cur.recvLength > 0) {
        out[0]->recv(recvRank, slot, cur.recvOffset, cur.recvLength);
      }
      if (cur.sendLength > 0) {
        out[0]->send(sendRank, slot, cur.sendOffset, cur.sendLength);
        // Broadcast first segments to outputs buffers.
        if (i < numSegmentsPerRank) {
          broadcastOutputs(cur.sendOffset, cur.sendLength);
        }
      }
    }
  }
}

// For a given context size and desired group size, compute the actual group
// size per step. Note that the group size per step is n for all steps, only
// if n^(#steps) == size. Otherwise, the final group size is != n.
std::vector<size_t> computeGroupSizePerStep(size_t size, const size_t n) {
  std::vector<size_t> result;
  GLOO_ENFORCE_GT(n, 1);
  while (size % n == 0) {
    result.push_back(n);
    size /= n;
  }
  if (size > 1) {
    result.push_back(size);
  }
  return result;
}

// The bcube algorithm implements a hypercube-like strategy for reduction. The
// constraint is that the number of processes can be factorized. If the minimum
// component in the factorization is 2, and the number of processes is equal to
// a power of 2, the algorithm is identical to recursive halving/doubling. The
// number of elements in the factorization determines the number of steps of the
// algorithm. Each element of the factorization determines the number of
// processes each process communicates with at that particular step of the
// algorithm. If the number of processes is not factorizable, the algorithm is
// identical to a direct reduce-scatter followed by allgather.
//
// For example, if #processes == 8, and we factorize as 4 * 2, the algorithm
// runs in 2 steps. In the first step, 2 groups of 4 processes exchange data
// such that all processes have 1/4th of the partial result (with process 0
// having the first quarter, 1 having the second quarter, and so forth). In the
// second step, 4 groups of 2 processes exchange their partial result such that
// all processes have 1/8th of the result. Then, the same factorization is
// followed in reverse to perform an allgather.
//
void bcube(
    const detail::AllreduceOptionsImpl& opts,
    ReduceRangeFunction reduceInputs,
    BroadcastRangeFunction broadcastOutputs) {
  const auto& context = opts.context;
  const auto slot = Slot::build(kAllreduceSlotPrefix, opts.tag);
  const auto elementSize = opts.elementSize;
  auto& out = opts.out[0];

  constexpr auto n = 2;

  // Figure out the number of steps in this algorithm.
  const auto groupSizePerStep = computeGroupSizePerStep(context->size, n);

  struct group {
    // Distance between peers in this group.
    size_t peerDistance;

    // Segment that this group is responsible for reducing.
    size_t bufferOffset;
    size_t bufferLength;

    // The process ranks that are a member of this group.
    std::vector<size_t> ranks;

    // Upper bound of the length of the chunk that each process has the
    // reduced values for by the end of the reduction for this group.
    size_t chunkLength;

    // Chunk within the segment that this process is responsible for reducing.
    size_t myChunkOffset;
    size_t myChunkLength;
  };

  // Compute the details of a group at every algorithm step.
  // We keep this in a vector because we iterate through it in forward order in
  // the reduce/scatter phase and in backward order in the allgather phase.
  std::vector<struct group> groups;
  {
    struct group group;
    group.peerDistance = 1;
    group.bufferOffset = 0;
    group.bufferLength = opts.elements;
    for (const size_t groupSize : groupSizePerStep) {
      const size_t groupRank = (context->rank / group.peerDistance) % groupSize;
      const size_t baseRank = context->rank - (groupRank * group.peerDistance);
      group.ranks.reserve(groupSize);
      for (size_t i = 0; i < groupSize; i++) {
        group.ranks.push_back(baseRank + i * group.peerDistance);
      }

      // Compute the length of the chunk we're exchanging at this step.
      group.chunkLength = ((group.bufferLength + (groupSize - 1)) / groupSize);

      // This process is computing the reduction of the chunk positioned at
      // <rank>/<size> within the current segment.
      group.myChunkOffset =
          group.bufferOffset + (groupRank * group.chunkLength);
      group.myChunkLength = std::min(
          size_t(group.chunkLength),
          size_t(std::max(
              int64_t(0),
              int64_t(group.bufferLength) -
                  int64_t(groupRank * group.chunkLength))));

      // Store a const copy of this group in the vector.
      groups.push_back(group);

      // Initialize with updated peer distance and segment offset and length.
      struct group nextGroup;
      nextGroup.peerDistance = group.peerDistance * groupSize;
      nextGroup.bufferOffset = group.myChunkOffset;
      nextGroup.bufferLength = group.myChunkLength;
      std::swap(group, nextGroup);
    }
  }

  // The chunk length is rounded up, so the maximum scratch space we need
  // might be larger than the size of the output buffer. Compute the maximum
  size_t bufferLength = opts.elements;
  for (const auto& group : groups) {
    bufferLength =
        std::max(bufferLength, group.ranks.size() * group.chunkLength);
  }

  // Allocate scratch space to receive data from peers.
  const size_t bufferSize = bufferLength * elementSize;
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[bufferSize]);
  std::unique_ptr<transport::UnboundBuffer> tmp =
      context->createUnboundBuffer(buffer.get(), bufferSize);

  // Reduce/scatter.
  for (size_t step = 0; step < groups.size(); step++) {
    const auto& group = groups[step];

    // Issue receive operations for chunks from peers.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto src = group.ranks[i];
      if (src == context->rank) {
        continue;
      }
      tmp->recv(
          src,
          slot,
          i * group.chunkLength * elementSize,
          group.myChunkLength * elementSize);
    }

    // Issue send operations for local chunks to peers.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto dst = group.ranks[i];
      if (dst == context->rank) {
        continue;
      }
      const size_t currentChunkOffset =
          group.bufferOffset + i * group.chunkLength;
      const size_t currentChunkLength = std::min(
          size_t(group.chunkLength),
          size_t(std::max(
              int64_t(0),
              int64_t(group.bufferLength) - int64_t(i * group.chunkLength))));
      // Compute the local reduction only in the first step of the algorithm.
      // In subsequent steps, we already have a partially reduced result.
      if (step == 0) {
        reduceInputs(
            currentChunkOffset * elementSize, currentChunkLength * elementSize);
      }
      out->send(
          dst,
          slot,
          currentChunkOffset * elementSize,
          currentChunkLength * elementSize);
    }

    // Wait for send and receive operations to complete.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto peer = group.ranks[i];
      if (peer == context->rank) {
        continue;
      }
      tmp->waitRecv();
      out->waitSend();
    }

    // In the first step, prepare the chunk this process is responsible for
    // with the reduced version of its inputs (if multiple are specified).
    if (step == 0) {
      reduceInputs(
          group.myChunkOffset * elementSize, group.myChunkLength * elementSize);
    }

    // Reduce chunks from peers.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto src = group.ranks[i];
      if (src == context->rank) {
        continue;
      }
      opts.reduce(
          static_cast<uint8_t*>(out->ptr) + (group.myChunkOffset * elementSize),
          static_cast<const uint8_t*>(out->ptr) +
              (group.myChunkOffset * elementSize),
          static_cast<const uint8_t*>(tmp->ptr) +
              (i * group.chunkLength * elementSize),
          group.myChunkLength);
    }
  }

  // There is one chunk that contains the final result and this chunk
  // can already be broadcast locally to out[1..N], if applicable.
  // Doing so means we only have to broadcast locally to out[1..N] all
  // chunks as we receive them from our peers during the allgather phase.
  {
    const auto& group = groups.back();
    broadcastOutputs(
        group.myChunkOffset * elementSize, group.myChunkLength * elementSize);
  }

  // Allgather.
  for (auto it = groups.rbegin(); it != groups.rend(); it++) {
    const auto& group = *it;

    // Issue receive operations for reduced chunks from peers.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto src = group.ranks[i];
      if (src == context->rank) {
        continue;
      }
      const size_t currentChunkOffset =
          group.bufferOffset + i * group.chunkLength;
      const size_t currentChunkLength = std::min(
          size_t(group.chunkLength),
          size_t(std::max(
              int64_t(0),
              int64_t(group.bufferLength) - int64_t(i * group.chunkLength))));
      out->recv(
          src,
          slot,
          currentChunkOffset * elementSize,
          currentChunkLength * elementSize);
    }

    // Issue send operations for reduced chunk to peers.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto dst = group.ranks[i];
      if (dst == context->rank) {
        continue;
      }
      out->send(
          dst,
          slot,
          group.myChunkOffset * elementSize,
          group.myChunkLength * elementSize);
    }

    // Wait for operations to complete.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto peer = group.ranks[i];
      if (peer == context->rank) {
        continue;
      }
      out->waitRecv();
      out->waitSend();
    }

    // Broadcast result to multiple output buffers, if applicable.
    for (size_t i = 0; i < group.ranks.size(); i++) {
      const auto peer = group.ranks[i];
      if (peer == context->rank) {
        continue;
      }
      const size_t currentChunkOffset =
          group.bufferOffset + i * group.chunkLength;
      const size_t currentChunkLength = std::min(
          size_t(group.chunkLength),
          size_t(std::max(
              int64_t(0),
              int64_t(group.bufferLength) - int64_t(i * group.chunkLength))));
      broadcastOutputs(
          currentChunkOffset * elementSize, currentChunkLength * elementSize);
    }
  }
}

} // namespace

void allreduce(const AllreduceOptions& opts) {
  allreduce(opts.impl_);
}

namespace {
// states for collectives
enum coll_state {
  coll_begin = 0,
  coll_allreduce_naive__copy_in_done,
  coll_allreduce_naive__reduce_done,
  // alternative state when allreduce is working on alternative buffer
  // of the double buffer.
  coll_alt1_allreduce_naive__copy_in_done,
  coll_alt2_allreduce_naive__copy_in_done,
  coll_alt1_allreduce_naive__reduce_done,
};

// SHM building blocks
struct SharedData {
  const char* name;
  int descriptor;
  void* bytes;
  size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    data->name = name;
    data->descriptor = d;
    data->bytes = bytes;
    data->nbytes = nbytes;
  } else {
    if (errno != ENOENT) {
      // don't print if shm can not be found because we want to loop over from
      // caller again until the other ranks created the shm
      printf("shared_open %s failed, errno=%d\n", name, errno);
    }
    data->descriptor = -1;
  }
}

void shared_create(
    SharedData* data,
    const char* name,
    void* bytes,
    size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    if (nbytes = write(d, bytes, nbytes)) {
      shared_open(data, name, nbytes);
    }
  } else {
    printf("shared_create %s failed\n", name);
  }
}

static int world_rank = -1;
static int world_size = -1;
static bool is_initialized = false;

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
  enum coll_state states[2]; // idx=0 -- state for symmetric_naive_all_reduce
                             // idx=1 -- state for distributed_naive_all_reduce
  // double buffer to avoid syncing between rounds
  // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for
  // symmetric_naive_all_reduce after that : buffer for
  // distributed_naive_all_reduce
  char buffer[2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) \
  2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE

struct allreduce_workspace** workspace;

// buffer for small messages, double buffer
char** symmetric_buffer[2];
// buffer for large messages, double buffer
char** distributed_buffer[2];

void wait_buffer_state_until_2(
    int index,
    enum coll_state state0,
    enum coll_state state1,
    int state_group) {
  volatile enum coll_state* state_ptr =
      &(workspace[index]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1)
      break;
  }
}

__m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src)
    __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src) {
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
}

__m512 cvt_fp16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_fp16_to_fp32(const __m256i src) {
  return _mm512_cvtph_ps(src);
}

inline __m256i cvt_fp32_to_fp16(const __m512 src)
    __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_fp16(const __m512 src) {
  return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

void reduce_bf16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_fp16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_fp32_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_all_buffers(
    int start_elements,
    int num_elements,
    AllreduceOptions::ScalarType scalar_type,
    int to_buffer_idx,
    char* to_buffer,
    char** buffers) {
  switch (scalar_type) {
    case AllreduceOptions::ScalarType::BFLOAT16:
      assert(!"BFloat16 not supported in gloo yet.");
      reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    case AllreduceOptions::ScalarType::HALF:
      reduce_fp16_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    case AllreduceOptions::ScalarType::FLOAT:
      reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    default:
      assert(!"Should not get here");
  }
}

#define CVT_ADD_BF16(x)                                                   \
  do {                                                                    \
    auto in##x##_val =                                                    \
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
    inout_val = _mm512_add_ps(inout_val, in##x##_val);                    \
  } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes
// processed each iteration depends on vector length.  256bit vector ==> 32
// bytes, 512bit vector ==> 64 bytes If you change implementation of
// reduce_bf16_buffers, etc. , check whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void reduce_bf16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val =
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
    switch (world_size) {
      case 16:
        CVT_ADD_BF16(15);
      case 15:
        CVT_ADD_BF16(14);
      case 14:
        CVT_ADD_BF16(13);
      case 13:
        CVT_ADD_BF16(12);
      case 12:
        CVT_ADD_BF16(11);
      case 11:
        CVT_ADD_BF16(10);
      case 10:
        CVT_ADD_BF16(9);
      case 9:
        CVT_ADD_BF16(8);
      case 8:
        CVT_ADD_BF16(7);
      case 7:
        CVT_ADD_BF16(6);
      case 6:
        CVT_ADD_BF16(5);
      case 5:
        CVT_ADD_BF16(4);
      case 4:
        CVT_ADD_BF16(3);
      case 3:
        CVT_ADD_BF16(2);
      case 2:
        CVT_ADD_BF16(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val =
              cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
          inout_val = _mm512_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_bf16(inout_val));
  }

  // process remaining part
  // todo: support bfloat16
  /*
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(at::BFloat16*)(buffers[j] + i);
    }
    *(at::BFloat16*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
    */
}

#define CVT_ADD_FP16(x)                                                   \
  do {                                                                    \
    auto in##x##_val =                                                    \
        cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
    inout_val = _mm512_add_ps(inout_val, in##x##_val);                    \
  } while (0)

void reduce_fp16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val =
        cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
    switch (world_size) {
      case 16:
        CVT_ADD_FP16(15);
      case 15:
        CVT_ADD_FP16(14);
      case 14:
        CVT_ADD_FP16(13);
      case 13:
        CVT_ADD_FP16(12);
      case 12:
        CVT_ADD_FP16(11);
      case 11:
        CVT_ADD_FP16(10);
      case 10:
        CVT_ADD_FP16(9);
      case 9:
        CVT_ADD_FP16(8);
      case 8:
        CVT_ADD_FP16(7);
      case 7:
        CVT_ADD_FP16(6);
      case 6:
        CVT_ADD_FP16(5);
      case 5:
        CVT_ADD_FP16(4);
      case 4:
        CVT_ADD_FP16(3);
      case 3:
        CVT_ADD_FP16(2);
      case 2:
        CVT_ADD_FP16(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val =
              cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
          inout_val = _mm512_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_fp16(inout_val));
  }

  

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float16 val =float16(0.0f);
    for (int j = 0; j < world_size; j++) {
      val += *(float16*)(buffers[j] + i);
    }
    *(float16*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

#define CVT_ADD_F32(x)                                            \
  do {                                                            \
    auto in##x##_val = _mm256_loadu_ps((float*)(buffers[x] + i)); \
    inout_val = _mm256_add_ps(inout_val, in##x##_val);            \
  } while (0)

void reduce_fp32_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 4;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val = _mm256_loadu_ps((float*)(buffers[0] + i));
    switch (world_size) {
      case 16:
        CVT_ADD_F32(15);
      case 15:
        CVT_ADD_F32(14);
      case 14:
        CVT_ADD_F32(13);
      case 13:
        CVT_ADD_F32(12);
      case 12:
        CVT_ADD_F32(11);
      case 11:
        CVT_ADD_F32(10);
      case 10:
        CVT_ADD_F32(9);
      case 9:
        CVT_ADD_F32(8);
      case 8:
        CVT_ADD_F32(7);
      case 7:
        CVT_ADD_F32(6);
      case 6:
        CVT_ADD_F32(5);
      case 5:
        CVT_ADD_F32(4);
      case 4:
        CVT_ADD_F32(3);
      case 3:
        CVT_ADD_F32(2);
      case 2:
        CVT_ADD_F32(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val = _mm256_loadu_ps((float*)(buffers[j] + i));
          inout_val = _mm256_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(float*)(buffers[j] + i);
    }
    *(float*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

void shm_initialize(int size, int rank, char* addr_string, char* port_string) {
  world_size = size;
  world_rank = rank;

  char shm_name_prefix[NAME_BUF_SIZE];
  char shm_name[NAME_BUF_SIZE];
  snprintf(
      shm_name_prefix,
      NAME_BUF_SIZE,
      "%s_%d_%s_%s",
      SHM_BUFFER_NAME,
      getuid(),
      addr_string,
      port_string);
  // create shared workspace for SHM based allreduce
  SharedData allreduce_buffer;
  // allocate workspace_buf for current rank
  struct allreduce_workspace* workspace_buf;
  struct allreduce_workspace* workspace_buf_other;
  workspace_buf =
      (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
  int written = snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, rank);
  if (written >= NAME_BUF_SIZE) {
    std::cout << "[warning]: written >= NAME_BUF_SIZE" << std::endl;
  }
  shared_create(
      &allreduce_buffer,
      shm_name,
      workspace_buf,
      sizeof(struct allreduce_workspace));
  workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
  workspace_buf->states[0] = coll_alt2_allreduce_naive__copy_in_done;
  workspace_buf->states[1] = coll_begin;

  // create the workspace pointer list
  workspace = (struct allreduce_workspace**)malloc(
      size * sizeof(struct allreduce_workspace*));
  symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
  symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

  // map shm of all ranks
  for (int i = 0; i < size; i++) {
    if (i != rank) {
        int written = snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, i);
        if (written >= NAME_BUF_SIZE) {
          std::cout << "[warning]: written >= NAME_BUF_SIZE" << std::endl;
        }
      // printf("open %s, %d\n", shm_name, rank);
      do {
        shared_open(
            &allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
      } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
      workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
      workspace[i] = workspace_buf_other;
    } else {
      workspace[i] = workspace_buf;
    }
    symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
    symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
    distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
    distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);
  }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes) {
  auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
  // process aligned part
#pragma omp parallel for
  for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
    auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
    _mm256_storeu_si256((__m256i*)((char*)to + i), val);
  }

  // process remaining part
  for (int i = aligned_bytes; i < n_bytes; i++) {
    *((char*)to + i) = *((char*)from + i);
  }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size)
                                     : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_size * slice_idx;
}

void symmetric_naive_all_reduce(
    char* data_ptr,
    AllreduceOptions::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el) {
  const int state_group = 0;
  static int current_buffer = 0;
  static int state_idx = 0;

  enum coll_state copy_current, copy_next;

  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      copy_next = coll_alt2_allreduce_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allreduce_naive__copy_in_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(
      symmetric_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until the other rank copy the buffer
    if (i != world_rank) {
      wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
    }
  }

  // each rank reduce the buffer independently so therre is no need for
  // synchronization afterward
  reduce_all_buffers(
      0,
      chunk_el,
      scalar_type,
      world_rank,
      data_ptr,
      symmetric_buffer[current_buffer]);

  // switch buffer
  current_buffer = 1 - current_buffer;
}

// naive allreduce distributed, each rank do naive reduce on its slice
void distributed_naive_reduce(
    char* data_ptr,
    AllreduceOptions::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el) {
  const int state_group = 1;
  static int current_buffer = 0;
  static int state_idx = 0;

  enum coll_state copy_current, copy_next, reduce_current;

  // similar to symmetric_naive_allreduce, but here we only need two sets of
  // states, because distributed naive reduce has two barriers in the algorithm
  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      reduce_current = coll_allreduce_naive__reduce_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      reduce_current = coll_alt1_allreduce_naive__reduce_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 2;

  int data_size = chunk_size / chunk_el;
  parallel_memcpy(
      distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != world_rank)
      wait_buffer_state_until_2(i, copy_current, reduce_current, state_group);
  }

  // reduce scatter
  reduce_all_buffers(
      slice_el_start(chunk_el, world_rank),
      slice_size(chunk_el, world_rank),
      scalar_type,
      world_rank,
      distributed_buffer[current_buffer][world_rank],
      distributed_buffer[current_buffer]);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = reduce_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks reduce the buffer
    if (i != world_rank)
      wait_buffer_state_until_2(i, reduce_current, copy_next, state_group);
  }

  for (int i = 0; i < world_size; i++) {
    int rank = (i + world_rank) % world_size;
    parallel_memcpy(
        slice_data(data_ptr, chunk_el, data_size, rank),
        slice_data(
            distributed_buffer[current_buffer][rank],
            chunk_el,
            chunk_size / chunk_el,
            rank),
        slice_size(chunk_el, rank) * data_size);
  }

  current_buffer = 1 - current_buffer;
}


void shm(const detail::AllreduceOptionsImpl& opts) {
  if (!is_initialized) {
    int size = std::stoi(std::getenv("PMI_SIZE"));
    int rank = std::stoi(std::getenv("PMI_RANK"));

    world_size = size;
    world_rank = rank;
    is_initialized = true;

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) {
        addr_string = "";
    }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) {
        port_string = "";
    }
    // std::cout << "size: " << size << std::endl;
    // std::cout << "rank: " << rank << std::endl;
    // std::cout << "addr_string: " << addr_string << std::endl;
    // std::cout << "port_string: " << port_string << std::endl;
    shm_initialize(size, rank, addr_string, port_string);
  }

  const size_t data_size = opts.elements * opts.elementSize;
  const std::vector<std::unique_ptr<transport::UnboundBuffer>>& out = opts.out;
  void* data = out[0].get()->ptr;
  // todo: set scalar type

    for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
        auto data_ptr = ((char*)(data) + offset);
        size_t chunk_size =
            data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
        size_t chunk_el = chunk_size / (data_size / opts.elements);
        if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
        symmetric_naive_all_reduce(
            data_ptr, opts.scalarType, chunk_size, chunk_el);
        } else {
        distributed_naive_reduce(
            data_ptr, opts.scalarType, chunk_size, chunk_el);
        }
  }

}

} // namespace


} // namespace gloo
