#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/prior_rbox_layer.hpp"

namespace caffe {

template <typename Dtype>
void PriorRBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PriorRBoxParameter& prior_rbox_param =
      this->layer_param_.prior_rbox_param();
  regress_size_ = prior_rbox_param.regress_size();
  regress_angle_ = prior_rbox_param.regress_angle();
  num_param_ = 2;
  if (regress_size_) num_param_ += 2;
  if (regress_angle_) num_param_ ++;
  CHECK_EQ(prior_rbox_param.prior_widths_size(), prior_rbox_param.prior_heights_size());
  for (int i = 0; i < prior_rbox_param.prior_widths_size(); i++)
  {
  	prior_widths_.push_back(prior_rbox_param.prior_widths(i));
  	prior_heights_.push_back(prior_rbox_param.prior_heights(i));
  }
  if (!regress_size_)
  {
  	CHECK_EQ(prior_widths_.size(), 1);
  	prior_width_ = prior_widths_[0];
  	prior_height_ = prior_heights_[0];
  }
  if (regress_angle_)
  	for (int i=0;i<prior_rbox_param.rotated_angles_size();i++)
 	 {
		  rotated_angles_.push_back(prior_rbox_param.rotated_angles(i));
 	 }
  else rotated_angles_.push_back(0);
  num_priors_ = rotated_angles_.size() * prior_widths_.size();
  clip_ = prior_rbox_param.clip();
  if (prior_rbox_param.variance_size() > 1) {
    // Must and only provide $num_param_ variance.
    CHECK_EQ(prior_rbox_param.variance_size(), num_param_);
    for (int i = 0; i < prior_rbox_param.variance_size(); ++i) {
      CHECK_GT(prior_rbox_param.variance(i), 0);
      variance_.push_back(prior_rbox_param.variance(i));
    }
  } else if (prior_rbox_param.variance_size() == 1) {
    CHECK_GT(prior_rbox_param.variance(0), 0);
    variance_.push_back(prior_rbox_param.variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }

  if (prior_rbox_param.has_img_h() || prior_rbox_param.has_img_w()) {
    CHECK(!prior_rbox_param.has_img_size())
        << "Either img_size or img_h/img_w should be specified; not both.";
    img_h_ = prior_rbox_param.img_h();
    CHECK_GT(img_h_, 0) << "img_h should be larger than 0.";
    img_w_ = prior_rbox_param.img_w();
    CHECK_GT(img_w_, 0) << "img_w should be larger than 0.";
  } else if (prior_rbox_param.has_img_size()) {
    const int img_size = prior_rbox_param.img_size();
    CHECK_GT(img_size, 0) << "img_size should be larger than 0.";
    img_h_ = img_size;
    img_w_ = img_size;
  } else {
    img_h_ = 0;
    img_w_ = 0;
  }

  if (prior_rbox_param.has_step_h() || prior_rbox_param.has_step_w()) {
    CHECK(!prior_rbox_param.has_step())
        << "Either step or step_h/step_w should be specified; not both.";
    step_h_ = prior_rbox_param.step_h();
    CHECK_GT(step_h_, 0.) << "step_h should be larger than 0.";
    step_w_ = prior_rbox_param.step_w();
    CHECK_GT(step_w_, 0.) << "step_w should be larger than 0.";
  } else if (prior_rbox_param.has_step()) {
    const float step = prior_rbox_param.step();
    CHECK_GT(step, 0) << "step should be larger than 0.";
    step_h_ = step;
    step_w_ = step;
  } else {
    step_h_ = 0;
    step_w_ = 0;
  }

  offset_ = prior_rbox_param.offset();
  
}

template <typename Dtype>
void PriorRBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);
  // Since all images in a batch has same height and width, we only need to
  // generate one set of priors which can be shared across all images.
  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  top_shape[2] = layer_width * layer_height * num_priors_ * num_param_; //each prior box has three parameters: cx,cy and angle
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PriorRBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = bottom[1]->width();
    img_height = bottom[1]->height();
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim = layer_height * layer_width * num_priors_ * num_param_;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset_) * step_w; //offset is default set to 0.5
      float center_y = (h + offset_) * step_h;
      //float box_width, box_height;
      //box_width = prior_width_;
	    //box_height = prior_height_;// box_width and box_height are not used recently
	    for (int i=0;i<rotated_angles_.size();i++){
	      for (int j=0;j<prior_widths_.size();j++){
		    // x_center
		    top_data[idx++] = center_x  / img_width;
		    // y_center
		    top_data[idx++] = center_y  / img_height;
		    if (regress_size_)
		    {
		    	top_data[idx++] = prior_widths_[j];
		    	top_data[idx++] = prior_heights_[j];
		    }
		    // angle
		    if (regress_angle_)
		    	top_data[idx++] = rotated_angles_[i]; 
		  }
		}
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }
  // set the variance.
  top_data += top[0]->offset(0, 1);
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } 
  else {
    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < num_param_; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PriorRBoxLayer);
REGISTER_LAYER_CLASS(PriorRBox);

}  // namespace caffe
