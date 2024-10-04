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
    TensorDataset<DType, LType> *Tensor_dataset;
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    int total_batches;
    /*TODO: add more member variables to support the iteration*/
public:
    DataLoader(Dataset<DType, LType> *ptr_dataset,
               int batch_size,
               bool shuffle = true,
               bool drop_last = false)
    {
        /*TODO: Add your code to do the initialization */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->Tensor_dataset = dynamic_cast<TensorDataset<DType, LType> *>(ptr_dataset);
        total_batches = Tensor_dataset->len() / batch_size;
        if (shuffle)
        {
            doShuffle();
        }
    }

    void doShuffle()
    {
        auto data = Tensor_dataset->getData();
        auto label = Tensor_dataset->getLabel();

        auto index = xt::arange<int>(0, data.shape()[0]);

        xt::random::default_engine_type engine(0);
        xt::random::shuffle(index, engine);

        auto shuffle_data = xt::empty<DType>(data.shape());
        auto shuffle_label = xt::empty<LType>(label.shape());

        if (data.dimension() != 0)
        {
            for (int i = 0; i < index.size(); i++)
            {
                xt::view(shuffle_data, i, xt::all()) = xt::view(data, index[i], xt::all());
            }
        }
        else
        {
            shuffle_data = data;
        }
        if (label.dimension() != 0)
        {
            for (int i = 0; i < index.size(); i++)
            {
                xt::view(shuffle_label, i, xt::all()) = xt::view(label, index[i], xt::all());
            }
        }
        else
        {
            shuffle_label = label;
        }
        Tensor_dataset->set_data(shuffle_data);
        Tensor_dataset->set_label(shuffle_label);
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
            int batch_size = ptr_loader->batch_size;
            bool drop_last = ptr_loader->drop_last;
            int remainder = ptr_loader->Tensor_dataset->len() % batch_size;
            int total_batches = ptr_loader->Tensor_dataset->len() / batch_size;
            if (cursor >= total_batches)
            {
                throw;
            }
            if (ptr_loader->Tensor_dataset->len() < batch_size)
            {
                throw;
            }
            xt::xarray<DType> data;
            xt::xarray<LType> label;
            for (int i = 0; i < total_batches; i++)
            {
                int start = i * batch_size;
                int end = batch_size * (i + 1);

                if (drop_last && i == total_batches - 1)
                {
                    if (ptr_loader->Tensor_dataset->getData().dimension() == 0)
                    {
                        data = ptr_loader->Tensor_dataset->getData();
                    }
                    else
                    {
                        data = xt::view(ptr_loader->Tensor_dataset->getData(), xt::range(start, end + remainder));
                    }
                    if (ptr_loader->Tensor_dataset->getLabel().dimension() == 0)
                    {
                        label = ptr_loader->Tensor_dataset->getLabel();
                    }
                    else
                    {
                        label = xt::view(ptr_loader->Tensor_dataset->getLabel(), xt::range(start, end + remainder));
                    }
                }
                else
                {
                    if (ptr_loader->Tensor_dataset->getData().dimension() == 0)
                    {
                        data = ptr_loader->Tensor_dataset->getData();
                    }
                    else
                    {
                        data = xt::view(ptr_loader->Tensor_dataset->getData(), xt::range(start, end));
                    }
                    if (ptr_loader->Tensor_dataset->getLabel().dimension() == 0)
                    {
                        label = ptr_loader->Tensor_dataset->getLabel();
                    }
                    else
                    {
                        label = xt::view(ptr_loader->Tensor_dataset->getLabel(), xt::range(start, end));
                    }
                }
            }
            return Batch<DType, LType>(data, label);
        }

        bool operator==(const Iterator &iterator) const
        {
            return cursor == iterator.cursor;
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
            Iterator iterator = *this;
            ++*this;
            return iterator;
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