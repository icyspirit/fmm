
#ifndef DISTRIBUTED_HPP
#define DISTRIBUTED_HPP


#include "mpi_util.hpp"
#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <cstring>


template<typename I>
inline I begin(I n, int size, int rank) noexcept
{
    return n*rank/size;
}


template<typename I>
inline I end(I n, int size, int rank) noexcept
{
    return n*(rank + 1)/size;
}


template<typename I>
inline I local(I n, int size, int rank) noexcept
{
    return end(n, size, rank) - begin(n, size, rank);
}


inline void allgatherv_inplace(int count, MPI_Datatype type, void* recvbuf, MPI_Comm comm) noexcept
{
    int size;
    MPI_Comm_size(comm, &size);

    int recvcounts[size];
    int displs[size];

    MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    displs[0] = 0;
    for (int rank=1; rank<size; ++rank) {
        displs[rank] = displs[rank - 1] + recvcounts[rank - 1];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf, recvcounts, displs, type, comm);
}


template<typename T>
class svector {
private:
    const MPI_Comm _comm;
    MPI_Win _window = MPI_WIN_NULL;

    bool _root = false;
    T* _data = nullptr;
    size_t _size = 0;

public:
    svector(MPI_Comm comm):
        _comm{comm}
    {
        int shm_rank;
        MPI_Comm_rank(_comm, &shm_rank);
        _root = shm_rank == 0;
    }

    svector(MPI_Comm comm, size_t size):
        svector(comm)
    {
        resize(size);
    }

    svector(svector&& other) noexcept:
        _comm{other._comm},
        _window{other._window},
        _root{other._root},
        _data{other._data},
        _size{other._size}
    {
        other._data = nullptr;
        other._window = MPI_WIN_NULL;
    }

    ~svector() noexcept
    {
        int flag;
        MPI_Finalized(&flag);
        if (!flag && _window != MPI_WIN_NULL) {
            MPI_Win_free(&_window);
        }
    }

    bool root() const noexcept
    {
        return _root;
    }

    T* data() noexcept
    {
        return _data;
    }

    const T* data() const noexcept
    {
        return _data;
    }

    size_t size() const noexcept
    {
        return _size;
    }

    void reserve(size_t size) noexcept
    {
        MPI_Win_allocate_shared(_root ? sizeof(T)*size : 0,
                                sizeof(T),
                                MPI_INFO_NULL,
                                _comm,
                                &_data,
                                &_window);

        MPI_Aint __size;
        int __disp_unit;
        MPI_Win_shared_query(_window, 0, &__size, &__disp_unit, &_data);
    }

    void resize(size_t size) noexcept
    {
        reserve(size);
        if (_root && _data) {
            std::fill_n(_data, size, T{});
        }
        sync();

        _size = size;
    }

    template<typename... Ts>
    void emplace_back(Ts&&... x) noexcept
    {
        new (_data + _size++) T(std::forward<Ts>(x)...);
    }

    template<typename I>
    inline T& operator [](I idx) noexcept
    {
        return _data[idx];
    }

    template<typename I>
    inline const T& operator [](I idx) const noexcept
    {
        return _data[idx];
    }

    inline void sync() const noexcept
    {
        MPI_Win_sync(_window);
        MPI_Barrier(_comm);
    }

    inline void sync_all() noexcept
    {
        MPI_Bcast(&_size, 1, get_mpi_type<size_t>(), 0, _comm);
        sync();
    }
};


#endif

