#ifndef CAFFE_PRIORRBOX_LAYER_HPP_
#define CAFFE_PRIORRBOX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Generate the prior boxes of designated sizes and aspect ratios across
 *        all dimensions @f$ (H \times W) @f$.
 *
 * Intended for use with MultiBox detection method to generate prior (template).
 *
 * NOTE: does not implement Backwards operation.
 *
 * Modified by LIU Lei
 */
template <typename Dtype>
class PriorRBoxLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides PriorRBoxParameter prior_box_param,
   *     with PriorBoxLayer options:
   *   - min_size (\b minimum box size in pixels. can be multiple. required!).
   *   - max_size (\b maximum box size in pixels. can be ignored or same as the
   *   # of min_size.).
   *   - aspect_ratio (\b optional aspect ratios of the boxes. can be multiple).
   *   - flip (\b optional bool, default true).
   *     if set, flip the aspect ratio.
   */
  explicit PriorRBoxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {} 
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PriorRBox"; }
  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @brief Generates prior boxes for a layer with specified parameters.
   *
   * @param bottom input Blob vector (at least 2)
   *   -# @f$ (N \times C \times H_i \times W_i) @f$
   *      the input layer @f$ x_i @f$
   *   -# @f$ (N \times C \times H_0 \times W_0) @f$
   *      the data layer @f$ x_0 @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 2 \times K*4) @f$ where @f$ K @f$ is the prior numbers
   *   By default, a box of aspect ratio 1 and min_size and a box of aspect
   *   ratio 1 and sqrt(min_size * max_size) are created.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  vector<float> min_sizes_;
  vector<float> max_sizes_;
  vector<float> aspect_ratios_;
  vector<float> rotated_angles_;
  vector<float> prior_widths_;
  vector<float> prior_heights_;
  float prior_width_;
  float prior_height_;
  bool regress_size_;
  bool regress_angle_;
  bool flip_;
  int num_priors_;
  bool clip_;
  vector<float> variance_;
  int num_param_;

  int img_w_;
  int img_h_;
  float step_w_;
  float step_h_;

  float offset_;
};

}  // namespace caffe

#endif  // CAFFE_PRIORBOX_LAYER_HPP_
