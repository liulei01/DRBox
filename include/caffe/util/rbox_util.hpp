#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_RBOX_UTIL_H_
#define CAFFE_UTIL_RBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

typedef EmitConstraint_EmitType EmitType;
typedef PriorRBoxParameter_CodeType CodeType;
typedef MultiRBoxLossParameter_MatchType MatchType;
typedef MultiRBoxLossParameter_LocLossType LocLossType;
typedef MultiRBoxLossParameter_ConfLossType ConfLossType;
typedef MultiRBoxLossParameter_MiningType MiningType;
typedef map<int, vector<NormalizedRBox> > LabelRBox;

struct Line
{
	int crossnum;//0:ignore; -1:all inner point; 2:two crossing point; 1:one crossing point
	int p1;//index of the start point
	int p2;//index of the end point
	int d[2][2];//the index of the start point after division
	float length;//the length after division
};


template <typename Dtype>
void GetGroundTruthR(const Dtype* gt_data, const int num_gt, const int background_label_id, 
	map<int, vector<NormalizedRBox> >* all_gt_rboxes);

template <typename Dtype>
void GetPriorRBoxes(const Dtype* prior_data, const int num_priors,
	const bool regress_angle, const bool regress_size,
	const float width, const float height,
	vector<NormalizedRBox>* prior_rboxes,
	vector<vector<float> >* prior_variances);

template <typename Dtype>
void GetLocPredictionsR(const Dtype* loc_data, const int num,
	const int num_preds,  const bool regress_angle, const bool regress_size,
	vector<LabelRBox>* loc_preds);
	
float JaccardOverlapR(const NormalizedRBox& rbox1, const NormalizedRBox& rbox2);
float JaccardOverlapRR(const NormalizedRBox& rbox1, const NormalizedRBox& rbox2);
float JaccardOverlapR(const NormalizedRBox& rbox1, const NormalizedRBox& rbox2, 
	const float width, const float height);
float JaccardOverlapRR(const NormalizedRBox& rbox1, const NormalizedRBox& rbox2,
	const float width, const float height);
template <typename Dtype>
Dtype JaccardOverlapR(const Dtype* rbox1, const Dtype* rbox2);

void MatchRBox(const vector<NormalizedRBox>& gt_rboxes,
	const vector<NormalizedRBox>& pred_rboxes, const int label,
	const MatchType match_type, const float overlap_threshold,
	const bool ignore_cross_boundary_rbox,
	vector<int>* match_indices, vector<float>* match_overlaps);

void MatchRBox(const vector<NormalizedRBox>& gt_rboxes,
	const vector<NormalizedRBox>& pred_rboxes, const int label,
	const MatchType match_type, const float overlap_threshold,
	const bool ignore_cross_boundary_rbox,
	vector<int>* match_indices, vector<float>* match_overlaps,
	const float prior_width, const float prior_height) ;

void FindMatchesR(const vector<LabelRBox>& all_loc_preds,
		const map<int, vector<NormalizedRBox> >& all_gt_rboxes,
		const vector<NormalizedRBox>& prior_rboxes,
		const vector<vector<float> >& prior_variances,
		const MultiRBoxLossParameter& multirbox_loss_param,
		vector<map<int, vector<float> > >* all_match_overlaps,
		vector<map<int, vector<int> > >* all_match_indices);

int CountNumMatchesR(const vector<map<int, vector<int> > >& all_match_indices,
	const int num);

template <typename Dtype>
void ComputeConfLossR(const Dtype* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	const int background_label_id, const ConfLossType loss_type,
	const vector<map<int, vector<int> > >& all_match_indices,
	const map<int, vector<NormalizedRBox> >& all_gt_rboxes,
	vector<vector<float> >* all_conf_loss);

template <typename Dtype>
void MineHardExamplesR(const Blob<Dtype>& conf_blob,
	const vector<LabelRBox>& all_loc_preds,
	const map<int, vector<NormalizedRBox> >& all_gt_rboxes,
	const vector<NormalizedRBox>& prior_rboxes,
	const vector<vector<float> >& prior_variances,
	const vector<map<int, vector<float> > >& all_match_overlaps,
	const MultiRBoxLossParameter& multirbox_loss_param,
	int* num_matches, int* num_negs,
	vector<map<int, vector<int> > >* all_match_indices,
	vector<vector<int> >* all_neg_indices);

void EncodeRBox(
	const NormalizedRBox& prior_rbox, const vector<float>& prior_variance,
	const CodeType code_type, const bool encode_variance_in_target,
	const NormalizedRBox& rbox, NormalizedRBox* encode_rbox,
	const bool regress_size, const bool regress_angle);

template <typename Dtype>
void EncodeLocPredictionR(const vector<LabelRBox>& all_loc_preds,
	const map<int, vector<NormalizedRBox> >& all_gt_rboxes,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<NormalizedRBox>& prior_rboxes,
	const vector<vector<float> >& prior_variances,
	const MultiRBoxLossParameter& multirbox_loss_param,
	Dtype* loc_pred_data, Dtype* loc_gt_data);

template <typename Dtype>
void EncodeConfPredictionR(const Dtype* conf_data, const int num,
	const int num_priors, const MultiRBoxLossParameter& multirbox_loss_param,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<vector<int> >& all_neg_indices,
	const map<int, vector<NormalizedRBox> >& all_gt_rboxes,
	Dtype* conf_pred_data, Dtype* conf_gt_data);

template <typename Dtype>
void GetLocPredictionsR(const Dtype* loc_data, const int num,
	const int num_preds_per_class, const int num_loc_classes,
	const bool regress_angle, const bool regress_size,
	const bool share_location, vector<LabelRBox>* loc_preds);

template <typename Dtype>
void GetConfidenceScoresR(const Dtype* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	vector<map<int, vector<float> > >* conf_preds);

void DecodeRBox(
	const NormalizedRBox& prior_rbox, const vector<float>& prior_variance,
	const CodeType code_type, const bool variance_encoded_in_target,
	const bool clip_rbox, const NormalizedRBox& rbox,
	const bool regress_size, const bool regress_angle,
	NormalizedRBox* decode_rbox);

void DecodeRBoxes(
	const vector<NormalizedRBox>& prior_rboxes,
	const vector<vector<float> >& prior_variances,
	const CodeType code_type, const bool variance_encoded_in_target,
	const bool clip_rbox, const vector<NormalizedRBox>& rboxes,
	const bool regress_size, const bool regress_angle,
	vector<NormalizedRBox>* decode_rboxes);

void DecodeRBoxesAll(const vector<LabelRBox>& all_loc_preds,
	const vector<NormalizedRBox>& prior_rboxes,
	const vector<vector<float> >& prior_variances,
	const int num, const bool share_location,
	const int num_loc_classes, const int background_label_id,
	const CodeType code_type, const bool variance_encoded_in_target,
	const bool clip, const bool regress_size, const bool regress_angle,
	vector<LabelRBox>* all_decode_rboxes);

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec);

void GetMaxScoreIndexR(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);

void ApplyNMSFastR(const vector<NormalizedRBox>& rboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices);

template <typename Dtype>
void GetRDetectionResults(const Dtype* det_data, const int num_det,
	const int background_label_id,
	map<int, map<int, vector<NormalizedRBox> > >* all_detections);

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);
}
#endif  // CAFFE_UTIL_BBOX_UTIL_H_
