//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"

#define MNN_OPEN_TIME_TRACE

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

using namespace MNN;
using namespace MNN::CV;
using std::cin;
using std::endl;
using std::cout;

static std::vector<std::string> class_names = {
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float NMS_THRES) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > NMS_THRES)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static cv::Mat draw_objects(const cv::Mat &rgb, const std::vector<Object> &objects) {

    cv::Mat image = rgb.clone();
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    return image;
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./frcnn.out model.mnn input.jpg\n");
        return 0;
    }
    const float NMS_THRES = 0.3f;
    const float CONF_THRES = 0.8f;
    const int num_category=int(class_names.size());

    timeval startime, endtime;
    cv::Mat raw_image;
    
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.numThread = 8;
    config.type = MNN_FORWARD_AUTO;
    auto session = net->createSession(config);
    //ScheduleConfig config;
    //config.numThread = 8;
    //config.type = MNN_FORWARD_AUTO;
    //auto session = net->createSession(config);
    auto input = net->getSessionInput(session, "data");
    std::vector<int> shape = input->shape();
    int input_H=shape[2];
    int input_W=shape[3];
    fprintf(stderr, "model input %d %d\n", input_H, input_W);
    //net->resizeTensor(input, shape);
    //net->resizeSession(session);
    auto input1 = net->getSessionInput(session, "im_info");
    //if (input1->elementSize() <= 4) {
    //    mnnNet->resizeTensor(input1, {1, 3, 600, 800});
    //    mnnNet->resizeSession(session);
    //}

    //Image Preprocessing
    gettimeofday(&startime, nullptr);
    auto inputPatch = argv[2];

    raw_image = cv::imread(inputPatch, 1);
    cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);

    if (raw_image.empty())
	{
           fprintf(stderr, "cv::imread %s failed\n", inputPatch);
           return -1;
    }

    int h = raw_image.rows;
    int w = raw_image.cols;
    fprintf(stderr, "w %d h %d\n", w, h);

    float scale = 1.f;
    if (w < h)
    {
        scale = (float)input_H / w;
        w = input_H;
        h = h * scale;
    }
    else
    {
        scale = (float)input_H / h;
        h = input_H;
        w = w * scale;
    }

    //ratio = std::min(1.0 * input_H / ori_height, 1.0 * input_W / ori_width);
    //int resize_height = int(ori_height * ratio);
    //int resize_width = int(ori_width * ratio);
    //odd number->pad size error
    //if (resize_height%2!=0) resize_height-=1;
    //if (resize_width%2!=0) resize_width-=1;

    //pad_W = int((input_W - resize_width) / 2);
    //pad_H = int((input_H - resize_height) / 2);

    cv::Mat in;
    cv::resize(raw_image, in, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    fprintf(stderr, "resize image w %d h %d\n", in.cols, in.rows);
    in.convertTo(in, CV_32FC3);

    //MNN::Tensor givenTensor(input, MNN::Tensor::CAFFE);
    MNN::Tensor givenTensor(input, MNN::Tensor::CAFFE);
    auto inputData = givenTensor.host<float>();
    //cv::Mat in_m = cv::Mat::zeros(h, w, CV_32FC3);
    for(int k = 0; k < 3; k++){
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            const auto src = in.at<cv::Vec3b>(i, j)[k];
            auto dst = 0.0;
            if(k == 0) dst = float(src) - 102.9801;
            if(k == 1) dst = float(src) - 115.9465;
            if(k == 2) dst = float(src) - 122.7717;
            //cout << "src " << float(src) << " dst " <<  dst << endl;
            inputData[k * w * h + i * w + j] = dst;
            //in_m.at<cv::Vec3b>(i, j)[k] = dst;
            }
        }
    }
    input->copyFromHostTensor(&givenTensor);
    //for(int i = 0; i < h; i++){
    //    for(int j = 0; j < w; j++){
    //        in.at<cv::Vec3b>(i, j)[0] = (float)(in.at<cv::Vec3b>(i, j)[0] - 102.9801);
    //        in.at<cv::Vec3b>(i, j)[1] = (float)(in.at<cv::Vec3b>(i, j)[1] - 115.9465);
    //        in.at<cv::Vec3b>(i, j)[2] = (float)(in.at<cv::Vec3b>(i, j)[2] - 122.7717);
    //        }
    //    }

    //for(int k = 0; k < 3; k++){
    //for(int i = 0; i < h; i++){
    //    for(int j = 0; j < w; j++){
    //        cout << "src " << in.at<cv::Vec3b>(i, j)[k] << endl;
    //        //in_m.at<cv::Vec3b>(i, j)[k] = dst;
    //        }
    //    }
    //}

    cout << "norm image" << endl;
    //cv::copyMakeBorder(resized_image, resized_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, pad);
    //resized_image = resized_image / 255.0f;

    // wrapping input tensor, convert nhwc to nchw
    //std::vector<int> dim{1, input_H, input_W, 3};
    //auto nhwc_Tensor = MNN::Tensor::create<float>(dim, NULL, MNN::Tensor::TENSORFLOW);
    //auto nhwc_data = nhwc_Tensor->host<float>();
    //auto nhwc_size = nhwc_Tensor->size();
    //::memcpy(nhwc_data, in.data, nhwc_size);
    //input->copyFromHostTensor(nhwc_Tensor);

    //MNN::Tensor givenTensor(input1, MNN::Tensor::CAFFE);
    //auto input1Data = givenTensor.host<float>();
    input1->host<float>()[0] = h;
    input1->host<float>()[1] = w;
    input1->host<float>()[2] = scale;
    //input1Data[0] = h;
    //input1Data[1] = w;
    //input1Data[2] = scale;
    //std::vector<int> dim1{1, im_info[0], im_info[1], scale};
    //auto im_info_Tensor = MNN::Tensor::create<float>(dim1, NULL, MNN::Tensor::TENSORFLOW);
    //auto im_info_data = nhwc_Tensor->host<float>();
    //auto im_info_size = nhwc_Tensor->size();
    //::memcpy(im_info_data, im_info.data, im_info_size);
    //input1->copyFromHostTensor(im_info_Tensor);
    //gettimeofday(&endtime, nullptr);
    cout << "preprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
    //Image Inference

    {
        gettimeofday(&startime, nullptr);
        net->runSession(session);
        gettimeofday(&endtime, nullptr);
        cout << "inferencetime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
    }
    //Image PostProcess
    {
        gettimeofday(&startime, nullptr);
        //auto conv_feats = net->getSessionOutput(session, "fire8/concat");
        //auto cf_host = std::make_shared<MNN::Tensor>(conv_feats, MNN::Tensor::CAFFE);
        //conv_feats->copyToHostTensor(cf_host.get());
        //auto cf_values = cf_host->host<float>();

        auto rois = net->getSessionOutput(session, "pps");
        auto rois_host = std::make_shared<MNN::Tensor>(rois, MNN::Tensor::CAFFE);
        rois->copyToHostTensor(rois_host.get());
        auto rois_values = rois_host->host<float>();
        auto rois_sizes = rois_host->elementSize();

        auto bbox_preds = net->getSessionOutput(session, "bbox_pred");
        auto pred_host = std::make_shared<MNN::Tensor>(bbox_preds, MNN::Tensor::CAFFE);
        bbox_preds->copyToHostTensor(pred_host.get());
        auto pred_values = pred_host->host<float>();
        auto pred_sizes =  pred_host->elementSize();

        auto cls_probs = net->getSessionOutput(session, "cls_prob");
        auto cls_host = std::make_shared<MNN::Tensor>(cls_probs, MNN::Tensor::CAFFE);
        cls_probs->copyToHostTensor(cls_host.get());
        auto cls_values = cls_host->host<float>();
        auto cls_sizes = cls_host->elementSize();
        fprintf(stderr, "get output finish\n");
        //cout << "conv shape " << cf_host->shape() << endl;
        //cout << "rois shape " << rois_host->shape()[0] << endl;
        //cout << "bbox shape " << pred_host->shape()[0] << endl;
        //cout << "cls shape " << cls_host->shape()[0] << endl;
        //cout << "rois shape " << rois_host->shape()[1] << endl;
        //cout << "bbox shape " << pred_host->shape()[1] << endl;
        //cout << "cls shape " << cls_host->shape()[1] << endl;

        cout << "rois size " << rois_sizes << endl;
        cout << "bbox size " << pred_sizes << endl;
        cout << "cls size " << cls_sizes << endl;

        //cout << "rois " << rois_values[0] << endl;
        //cout << "rois " << rois_values[1] << endl;
        //cout << "rois " << rois_values[2] << endl;
        //cout << "rois " << rois_values[3] << endl;
        //cout << "rois " << rois_values[4] << endl;
        //cout << "rois " << rois_values[5] << endl;

        //cout << "bbox " << pred_values[0] << endl;
        //cout << "bbox " << pred_values[1] << endl;
        //cout << "bbox " << pred_values[2] << endl;
        //cout << "bbox " << pred_values[3] << endl;

        //cout << "bbox " << pred_values[84] << endl;
        //cout << "bbox " << pred_values[85] << endl;
        //cout << "bbox " << pred_values[86] << endl;
        //cout << "bbox " << pred_values[87] << endl;

        //cout << "cls " << cls_values[0] << endl;
        //cout << "cls " << cls_values[21] << endl;
        //auto dimType = output->getDimensionType();
        //if (output->getType().code != halide_type_float) {
        //    dimType = Tensor::TENSORFLOW;
        //}

        //std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
        //output->copyToHostTensor(outputUser.get());
        //auto type = outputUser->getType();

        //auto size = outputUser->elementSize();
        //std::vector<float> tempValues(size);
        //if (type.code == halide_type_float) {
        //    auto values = outputUser->host<float>();
        //    for (int i = 0; i < size; ++i) {
        //        tempValues[i] = values[i];
        //    }
        //}

        //auto OUTPUT_NUM = outputUser->shape()[0];
        std::vector<std::vector<Object> > class_candidates;
        //std::vector<int> tempcls;

        for (int i = 0; i < 100; i++) {

            //int num_class = cls_values[i].w;
            class_candidates.resize(num_category);

            // find class id with highest score
            int label = 0;
            float score = 0.f;
            for (int j=0; j<num_category; j++)
            {
                float class_score = cls_values[j + i*num_category];
                if (class_score > score)
                {
                    label = j;
                    score = class_score;
                }
            }

            cout << "label " << label << " cls " << score << endl;
            // ignore background or low score
            if (label != 14 || score <= CONF_THRES)
                continue;

//             fprintf(stderr, "%d = %f\n", label, score);
            
            //auto roi = rois_values[i];
            //auto bbox_pred = pred_values[i];

            // unscale to image size
            float x1 = (rois_values[i * 5 + 1] / 2) / scale;
            float y1 = (rois_values[i * 5 + 2] / 2) / scale;
            float x2 = (rois_values[i * 5 + 3] / 2) / scale;
            float y2 = (rois_values[i * 5 + 4] / 2) / scale;

            cout << "rois org x1 " << rois_values[i * 5 + 1] << endl;
            cout << "rois org y1 " << rois_values[i * 5 + 2] << endl;
            cout << "rois org x2 " << rois_values[i * 5 + 3] << endl;
            cout << "rois org y2 " << rois_values[i * 5 + 4] << endl;

            cout << "rois scaled x1 " << rois_values[i * 5 + 1] / 2 << endl;
            cout << "rois scaled y1 " << rois_values[i * 5 + 2] / 2 << endl;
            cout << "rois scaled x2 " << rois_values[i * 5 + 3] / 2 << endl;
            cout << "rois scaled y2 " << rois_values[i * 5 + 4] / 2 << endl;

            cout << "rois x1 " << x1 << endl;
            cout << "rois y1 " << y1 << endl;
            cout << "rois x2 " << x2 << endl;
            cout << "rois y2 " << y2 << endl;

            float pb_w = x2 - x1 + 1;
            float pb_h = y2 - y1 + 1;

            // apply bbox regression
            float dx = pred_values[label * 4];
            float dy = pred_values[label * 4 + 1];
            float dw = pred_values[label * 4 + 2];
            float dh = pred_values[label * 4 + 3];

            float cx = x1 + pb_w * 0.5f;
            float cy = y1 + pb_h * 0.5f;

            float obj_cx = cx + pb_w * dx;
            float obj_cy = cy + pb_h * dy;

            float obj_w = pb_w * exp(dw);
            float obj_h = pb_h * exp(dh);

            float obj_x1 = obj_cx - obj_w * 0.5f;
            float obj_y1 = obj_cy - obj_h * 0.5f;
            float obj_x2 = obj_cx + obj_w * 0.5f;
            float obj_y2 = obj_cy + obj_h * 0.5f;

            // clip
            obj_x1 = std::max(std::min(obj_x1, (float)(raw_image.cols - 1)), 0.f);
            obj_y1 = std::max(std::min(obj_y1, (float)(raw_image.rows - 1)), 0.f);
            obj_x2 = std::max(std::min(obj_x2, (float)(raw_image.cols - 1)), 0.f);
            obj_y2 = std::max(std::min(obj_y2, (float)(raw_image.rows - 1)), 0.f);

            // append object
            Object obj;
            obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2-obj_x1+1, obj_y2-obj_y1+1);
            obj.label = label;
            obj.prob = score;

            class_candidates[label].push_back(obj);
            //auto prob = tempValues[i * (5+num_category) + 4];
            //auto maxcls = std::max_element(tempValues.begin() + i * (5+num_category) + 5, tempValues.begin() + i * (5+num_category) + (5+num_category));
            //auto clsidx = maxcls - (tempValues.begin() + i * (5+num_category) + 5);
            //auto score = prob * (*maxcls);
            //if (score < CONF_THRES) continue;
            //auto xmin = (tempValues[i * (5+num_category) + 0] - pad_W) / ratio;
            //auto xmax = (tempValues[i * (5+num_category) + 2] - pad_W) / ratio;
            //auto ymin = (tempValues[i * (5+num_category) + 1] - pad_H) / ratio;
            //auto ymax = (tempValues[i * (5+num_category) + 3] - pad_H) / ratio;

            //Object obj;
            //obj.rect = cv::Rect_<float>(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            //obj.label = clsidx;
            //obj.prob = score;
            //class_candidates[clsidx].push_back(obj);
        }

        std::vector<Object> objects;
        for (int i = 0; i < (int)class_candidates.size(); i++) {
            std::vector<Object> &candidates = class_candidates[i];

            qsort_descent_inplace(candidates);

            std::vector<int> picked;
            nms_sorted_bboxes(candidates, picked, NMS_THRES);

            for (int j = 0; j < (int)picked.size(); j++) {
                int z = picked[j];
                objects.push_back(candidates[z]);
            }
        }

        qsort_descent_inplace(objects);

        gettimeofday(&endtime, nullptr);
        cout << "postprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
        auto imgshow = draw_objects(raw_image, objects);
        //cv::imshow("w", imgshow);
        //cv::waitKey(-1);
        cv::imwrite("results.jpg", imgshow);
        return 0;
    }
}
