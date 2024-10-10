#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"

using namespace std;

template <typename DType, typename LType>
class DataLoader
{
public:
    class Iterator;

private:
    Dataset<DType, LType> *ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    xt::xarray<DType> shuffle_data;
    xt::xarray<LType> shuffle_label;
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::xarray<unsigned long> index;
    int total_batches;
    int m_seed;
    /*TODO: add more member variables to support the iteration*/
public:
    DataLoader(Dataset<DType, LType> *ptr_dataset,
               int batch_size,
               bool shuffle = true,
               bool drop_last = false, int seed = -1)
    {
        /*TODO: Add your code to do the initialization */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->m_seed = seed;
        // copy data and label from dataset
        this->total_batches = ptr_dataset->len() / batch_size;
        this->index = xt::arange(0, ptr_dataset->len());
        if (shuffle)
        {
            doShuffle();
        }

        shuffle_data = xt::empty<DType>(ptr_dataset->get_data_shape());
        shuffle_label = xt::empty<LType>(ptr_dataset->get_label_shape());

        if (shuffle_data.shape()[0] == 0)
        {
            DataLabel<DType, LType> data_label = ptr_dataset->getitem(0);
            shuffle_data = data_label.getData();
        }
        else
        {
            for (int i = 0; i < ptr_dataset->len(); i++)
            {
                DataLabel<DType, LType> data_label = ptr_dataset->getitem(index[i]);
                xt::view(shuffle_data, i) = data_label.getData();
            }
        }
        if (shuffle_label.shape()[0] == 0)
        {
            DataLabel<DType, LType> data_label = ptr_dataset->getitem(0);
            shuffle_label = data_label.getLabel();
        }
        else
        {
            for (int i = 0; i < ptr_dataset->len(); i++)
            {
                DataLabel<DType, LType> data_label = ptr_dataset->getitem(index[i]);
                xt::view(shuffle_label, i) = data_label.getLabel();
            }
        }
    }

    void doShuffle()
    {
        if (m_seed > 0)
        {
            xt::random::seed(m_seed);
            xt::random::shuffle(index);
        }
        else
        {
            xt::random::shuffle(index);
        }
    }
    virtual ~DataLoader()
    {
    }

    Iterator begin()
    {
        return Iterator(this, 0);
    }

    Iterator end()
    {
        return Iterator(this, total_batches);
    }
    class Iterator
    {
    private:
        DataLoader<DType, LType> *ptr_loader;
        int cursor;

    public:
        Iterator(DataLoader<DType, LType> *ptr_loader, int cursor) : ptr_loader(ptr_loader), cursor(cursor) {}
        Iterator &operator=(const Iterator &iterator)
        {
            cursor = iterator.cursor;
            ptr_loader = iterator.ptr_loader;
            return *this;
        }

        // Dereferencing overload
        Batch<DType, LType> operator*() const
        {
            xt::xarray<DType> data = ptr_loader->shuffle_data;
            xt::xarray<LType> label = ptr_loader->shuffle_label;
            xt::xarray<DType> batch_data;
            xt::xarray<LType> batch_label;
            int total_batches = ptr_loader->total_batches;

            if (cursor >= ptr_loader->total_batches)
            {
                throw;
            }
            int start = cursor * ptr_loader->batch_size;
            int end = (cursor + 1) * ptr_loader->batch_size;

            if (cursor == total_batches - 1)
            {
                if (ptr_loader->drop_last == true)
                    end = ptr_loader->ptr_dataset->len() - ptr_loader->ptr_dataset->len() % ptr_loader->batch_size;
                else if (ptr_loader->drop_last == false)
                    end = ptr_loader->ptr_dataset->len();
            }
            if (data.dimension() != 0)
            {
                batch_data = xt::view(data, xt::range(start, end));
            }
            else
            {
                batch_data = data;
            }
            if (label.dimension() != 0)
            {
                batch_label = xt::view(label, xt::range(start, end));
            }
            else
            {
                batch_label = label;
            }
            return Batch<DType, LType>(batch_data, batch_label);

            // int start = cursor * ptr_loader->batch_size;
            // int end = start + ptr_loader->batch_size;
            // if (cursor == ptr_loader->total_batch - 1)
            // {
            //   if (ptr_loader->drop_last == true)
            //     end = ptr_loader->ptr_dataset->len() - ptr_loader->ptr_dataset->len() % ptr_loader->batch_size;
            //   else if (ptr_loader->drop_last == false)
            //     end = ptr_loader->ptr_dataset->len();
            // }

            // if (ptr_loader->copied_data.dimension() == 0)
            //   data = ptr_loader->copied_data;
            // else
            //   data = xt::view(ptr_loader->copied_data, xt::range(start, end));

            // if (ptr_loader->copied_label.dimension() == 0)
            //   label = ptr_loader->copied_label;
            // else
            //   label = xt::view(ptr_loader->copied_label, xt::range(start, end));
            // return Batch<DType, LType>(data, label);
        }

        bool operator!=(const Iterator &iterator) const
        {
            return cursor != iterator.cursor;
        }

        // Prefix overload
        Iterator &operator++()
        {
            this->cursor++;
            return *this;
        }
        // Postfix overload
        Iterator operator++(int)
        {
            Iterator temp = *this;
            ++cursor;
            return temp;
        }
    };

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    /*TODO: Add your code here to support iteration on batch*/

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};

#endif /* DATALOADER_H */