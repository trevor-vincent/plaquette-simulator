#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "Error.hpp"
#include "GateFunctors.hpp"

template <class Precision> class CliffordStateKokkos {
  
  public:
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<Precision *>;

  using KokkosSizeTVector = Kokkos::View<size_t *>;
  using KokkosRangePolicy = Kokkos::RangePolicy<KokkosExecSpace>;

  using UnmanagedSizeTHostView =
    Kokkos::View<size_t *, Kokkos::HostSpace,
		 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using UnmanagedIntHostView =
    Kokkos::View<int *, Kokkos::HostSpace,
		 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  
};
