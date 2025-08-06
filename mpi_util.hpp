
#ifndef MPI_UTIL_HPP
#define MPI_UTIL_HPP


#include <mpi.h>
#include <cassert>
#include <cstddef>
#include <limits.h>
#include <stdexcept>


#define MPI_CHECK(X) \
do { \
int status = X; \
assert(status == 0); \
} while (0)


inline int get_size()
{
    static int size = MPI_UNDEFINED;
    if (size == MPI_UNDEFINED) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    return size;
}


inline int get_rank()
{
    static int rank = MPI_UNDEFINED;
    if (rank == MPI_UNDEFINED) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    return rank;
}


inline MPI_Comm get_shm_comm()
{
    static MPI_Comm shm_comm = MPI_UNDEFINED;
    if (shm_comm == MPI_UNDEFINED) {
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    }
    return shm_comm;
}


inline int get_shm_size()
{
    static int shm_size = MPI_UNDEFINED;
    if (shm_size == MPI_UNDEFINED) {
        MPI_Comm_size(get_shm_comm(), &shm_size);
    }
    return shm_size;
}


inline int get_shm_rank()
{
    static int shm_rank = MPI_UNDEFINED;
    if (shm_rank == MPI_UNDEFINED) {
        MPI_Comm_rank(get_shm_comm(), &shm_rank);
    }
    return shm_rank;
}


inline int get_node_size()
{
    static int node_size = MPI_UNDEFINED;
    if (node_size == MPI_UNDEFINED) {
        assert(get_size()%get_shm_size() == 0);
        node_size = get_size()/get_shm_size();
    }
    return node_size;
}


inline int get_node_rank()
{
    static int node_rank = MPI_UNDEFINED;
    if (node_rank == MPI_UNDEFINED) {
        assert(get_size()%get_shm_size() == 0);
        node_rank = get_rank()/get_shm_size();
    }
    return node_rank;
}


MPI_Comm get_split_comm(MPI_Comm comm, int color, int key)
{
    MPI_Comm new_comm;
    MPI_Comm_split(comm, color, key, &new_comm);
    return new_comm;
}


template<typename T>
MPI_Datatype get_mpi_type()
{
    throw std::runtime_error("Not implemented");
}


template<>
MPI_Datatype get_mpi_type<int>()
{
    return MPI_INT;
}


template<>
MPI_Datatype get_mpi_type<int64_t>()
{
    return MPI_INT64_T;
}


template<>
MPI_Datatype get_mpi_type<size_t>()
{
#if SIZE_MAX == UCHAR_MAX
    return MPI_UNSIGNED_CHAR;
#elif SIZE_MAX == USHRT_MAX
    return MPI_UNSIGNED_SHORT;
#elif SIZE_MAX == UINT_MAX
    return MPI_UNSIGNED;
#elif SIZE_MAX == ULONG_MAX
    return MPI_UNSIGNED_LONG;
#elif SIZE_MAX == ULLONG_MAX
    return MPI_UNSIGNED_LONG_LONG;
#else
   #error "What..?"
#endif
}


template<>
MPI_Datatype get_mpi_type<float>()
{
    return MPI_FLOAT;
}


template<>
MPI_Datatype get_mpi_type<double>()
{
    return MPI_DOUBLE;
}


#endif

