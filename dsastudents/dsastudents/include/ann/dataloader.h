/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

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
        TensorDataset<DType, LType> *ptr_dataset = dynamic_cast<TensorDataset<DType, LType> *>(ptr_dataset);
        xt::svector << Batch<DType, LType> * > batches;
        xt::xarray<DType> data;
        xt::xarray<LType> label;
        if (shuffle)
        {
            doShuffle();
        }
        generate_batches();
    }

    void doShuffle()
    {
        xt::xarray<DType> data = ptr_dataset->get_data();
        xt::xarray<LType> label = ptr_dataset->get_label();

        int index = xt::arange(ptr_dataset->len());

        xt::random::default_engine_type engine(0);
        xt::random::shuffle(index, engine);

        xt::xarray shuffle_data = xt::zeros<DType>(data.shape()[1]);
        xt::xarray shuffle_label = xt::zeros<LType>(label.shape()[1]);

        if (data.dimension() != 0)
        {
            for (int i = 0; i < index.size(); i++)
            {
                xt::view(shuffle_data, i, xt::all() == xt::view(data, index[i], xt::all()));
            }
        }
        if (label.dimension() != 0)
        {
            for (int i = 0; i < index.size(); i++)
            {
                xt::view(shuffle_label, i, xt::all() == xt::view(label, index[i], xt::all()));
            }
        }
        ptr_dataset->set_data(shuffle_data);
        ptr_dataset->set_label(shuffle_label);
    }

    void generate_batches()
    {
        remainder = ptr_dataset->len() % batch_size;
        total_batches = ptr_dataset->len() / batch_size;
        for (int i = 0; i < total_batches; i++)
        {
            if (droplast && i == total_batches - 1)
            {
                if (ptr_dataset->get_data()->dimension() == 0)
                {
                    data = ptr_dataset->get_data();
                }
                else
                {
                    data = xt::view(ptr_dataset->get_data(), xt::range(i * batch_size, batch_size * i + 30 + remainder));
                }
                if (ptr_dataset->get_label()->dimension() == 0)
                {
                    label = ptr_dataset->get_label();
                }
                else
                {
                    label = xt::view(ptr_dataset->get_label(), xt::range(i * batch_size, batch_size * i + 30 + remainder));
                }
                batches.push_back(new Batch<DType, LType>(data, label));
            }
            else
            {
                if (ptr_dataset->get_data()->dimension() == 0)
                {
                    data = ptr_dataset->get_data();
                }
                else
                {
                    data = xt::view(ptr_dataset->get_data(), xt::range(i * batch_size, batch_size * i + 30));
                }
                if (ptr_dataset->get_label()->dimension() == 0)
                {
                    label = ptr_dataset->get_label();
                }
                else
                {
                    label = xt::view(ptr_dataset->get_label(), xt::range(i * batch_size, batch_size * i + 30));
                }
                batches.push_back(new Batch<DType, LType>(data, label));
            }
        }
    }
    virtual ~DataLoader()
    {
        // destroy batches
        for (int i = 0; i < batches.size(); i++)
        {
            delete batches[i];
        }
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
            return *ptr_loader->batches[cursor];
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
