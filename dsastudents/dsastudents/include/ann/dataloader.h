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
        // shuffe the dataset
    }

    void generate_batches()
    {
        remainder = ptr_dataset->len() % batch_size;
        total_batches = ptr_dataset->len() / batch_size;
        for (int i = 0; i < total_batches; i++)
        {
            if (droplast && i == total_batches - 1)
            {
                if (ptr_dataset->get_data()->Dimension() == 0)
                {
                    data = ptr_dataset->get_data();
                }
                else
                {
                    data = xt::view(ptr_dataset->get_data(), xt::range(i * batch_size, batch_size * i + 30 + remainder));
                }
                if (ptr_dataset->get_label()->Dimension() == 0)
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
                if (ptr_dataset->get_data()->Dimension() == 0)
                {
                    data = ptr_dataset->get_data();
                }
                else
                {
                    data = xt::view(ptr_dataset->get_data(), xt::range(i * batch_size, batch_size * i + 30));
                }
                if (ptr_dataset->get_label()->Dimension() == 0)
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
    virtual ~DataLoader() {}

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
