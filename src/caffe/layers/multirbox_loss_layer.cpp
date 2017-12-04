#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/layers/multirbox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {

	template <typename Dtype>
	void MultiRBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			LossLayer<Dtype>::LayerSetUp(bottom, top);
			if (this->layer_param_.propagate_down_size() == 0) {
				this->layer_param_.add_propagate_down(true);
				this->layer_param_.add_propagate_down(true);
				this->layer_param_.add_propagate_down(false);
				this->layer_param_.add_propagate_down(false);
			}
			const MultiRBoxLossParameter& multirbox_loss_param =
				this->layer_param_.multirbox_loss_param();
			multirbox_loss_param_ = this->layer_param_.multirbox_loss_param();
			regress_angle_ = multirbox_loss_param.regress_angle();
			regress_size_ = multirbox_loss_param.regress_size();
			num_ = bottom[0]->num();
			LOG(INFO) << "MultiRBoxLoss";
			int tmp = 2;
			if (regress_angle_) tmp ++;
			if (regress_size_) tmp += 2;
			num_priors_ = bottom[2]->height() / tmp;
			LOG(INFO) << "num_param = "<<tmp;
			// Get other parameters.
			CHECK(multirbox_loss_param.has_num_classes()) << "Must provide num_classes.";
			num_classes_ = multirbox_loss_param.num_classes();
			CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
			share_location_ = multirbox_loss_param.share_location();
			loc_classes_ = share_location_ ? 1 : num_classes_;
			background_label_id_ = multirbox_loss_param.background_label_id();
			use_difficult_gt_ = multirbox_loss_param.use_difficult_gt();
			mining_type_ = multirbox_loss_param.mining_type();
			if (multirbox_loss_param.has_do_neg_mining()) {
				LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
				do_neg_mining_ = multirbox_loss_param.do_neg_mining();
				CHECK_EQ(do_neg_mining_,
					mining_type_ != MultiRBoxLossParameter_MiningType_NONE);
			}
			do_neg_mining_ = mining_type_ != MultiRBoxLossParameter_MiningType_NONE;
			if (regress_size_)
			{
				prior_width_ = -1;
				prior_height_ = -1;
			}
			else
			{
				prior_width_ = multirbox_loss_param.prior_width();
				prior_height_ = multirbox_loss_param.prior_height();
			}
			if (!this->layer_param_.loss_param().has_normalization() &&
				this->layer_param_.loss_param().has_normalize()) {
					normalization_ = this->layer_param_.loss_param().normalize() ?
LossParameter_NormalizationMode_VALID :
					LossParameter_NormalizationMode_BATCH_SIZE;
			} else {
				normalization_ = this->layer_param_.loss_param().normalization();
			}

			if (do_neg_mining_) {
				CHECK(share_location_)
					<< "Currently only support negative mining if share_location is true.";
			}

			vector<int> loss_shape(1, 1);
			// Set up localization loss layer.
			loc_weight_ = multirbox_loss_param.loc_weight();
			loc_loss_type_ = multirbox_loss_param.loc_loss_type();
			// fake shape.
			vector<int> loc_shape(1, 1);
			loc_shape.push_back(4);
			loc_pred_.Reshape(loc_shape);
			loc_gt_.Reshape(loc_shape);
			loc_bottom_vec_.push_back(&loc_pred_);
			loc_bottom_vec_.push_back(&loc_gt_);
			loc_loss_.Reshape(loss_shape);
			loc_top_vec_.push_back(&loc_loss_);
			if (loc_loss_type_ == MultiRBoxLossParameter_LocLossType_L2) {
				LayerParameter layer_param;
				layer_param.set_name(this->layer_param_.name() + "_l2_loc");
				layer_param.set_type("EuclideanLoss");
				layer_param.add_loss_weight(loc_weight_);
				loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
				loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
			} else if (loc_loss_type_ == MultiRBoxLossParameter_LocLossType_SMOOTH_L1) {
				LayerParameter layer_param;
				layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
				layer_param.set_type("SmoothL1Loss");
				layer_param.add_loss_weight(loc_weight_);
				loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
				loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
			} else {
				LOG(FATAL) << "Unknown localization loss type.";
			}
			// Set up confidence loss layer.
			conf_loss_type_ = multirbox_loss_param.conf_loss_type();
			conf_bottom_vec_.push_back(&conf_pred_);
			conf_bottom_vec_.push_back(&conf_gt_);
			conf_loss_.Reshape(loss_shape);
			conf_top_vec_.push_back(&conf_loss_);
			if (conf_loss_type_ == MultiRBoxLossParameter_ConfLossType_SOFTMAX) {
				CHECK_GE(background_label_id_, 0)
					<< "background_label_id should be within [0, num_classes) for Softmax.";
				CHECK_LT(background_label_id_, num_classes_)
					<< "background_label_id should be within [0, num_classes) for Softmax.";
				LayerParameter layer_param;
				layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
				layer_param.set_type("SoftmaxWithLoss");
				layer_param.add_loss_weight(Dtype(1.));
				layer_param.mutable_loss_param()->set_normalization(
					LossParameter_NormalizationMode_NONE);
				SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
				softmax_param->set_axis(1);
				// Fake reshape.
				vector<int> conf_shape(1, 1);
				conf_gt_.Reshape(conf_shape);
				conf_shape.push_back(num_classes_);
				conf_pred_.Reshape(conf_shape);
				conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
				conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
			} else if (conf_loss_type_ == MultiRBoxLossParameter_ConfLossType_LOGISTIC) {
				LayerParameter layer_param;
				layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
				layer_param.set_type("SigmoidCrossEntropyLoss");
				layer_param.add_loss_weight(Dtype(1.));
				// Fake reshape.
				vector<int> conf_shape(1, 1);
				conf_shape.push_back(num_classes_);
				conf_gt_.Reshape(conf_shape);
				conf_pred_.Reshape(conf_shape);
				conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
				conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
			} else {
				LOG(FATAL) << "Unknown confidence loss type.";
			}
	}

	template <typename Dtype>
	void MultiRBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			LossLayer<Dtype>::Reshape(bottom, top);
			num_ = bottom[0]->num();
			int tmp = 2;
			if (regress_angle_) tmp ++;
			if (regress_size_) tmp += 2;
			num_priors_ = bottom[2]->height() / tmp;
			num_gt_ = bottom[3]->height();
			CHECK_EQ(bottom[0]->num(), bottom[1]->num());
			CHECK_EQ(num_priors_ * loc_classes_ * tmp, bottom[0]->channels())
				<< "Number of priors must match number of location predictions.";
			CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
				<< "Number of priors must match number of confidence predictions.";
	}

	template <typename Dtype>
	void MultiRBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* loc_data = bottom[0]->cpu_data();
			const Dtype* conf_data = bottom[1]->cpu_data();
			const Dtype* prior_data = bottom[2]->cpu_data();
			const Dtype* gt_data = bottom[3]->cpu_data();
			// Retrieve all ground truth.
			map<int, vector<NormalizedRBox> > all_gt_rboxes;
			GetGroundTruthR(gt_data, num_gt_, background_label_id_, &all_gt_rboxes);


			// Retrieve all prior rboxes. It is same within a batch since we assume all
			// images in a batch are of same dimension.
			vector<NormalizedRBox> prior_rboxes;
			vector<vector<float> > prior_variances;
			GetPriorRBoxes(prior_data, num_priors_, regress_angle_, regress_size_, 
				prior_width_, prior_height_, &prior_rboxes, &prior_variances);

			// Retrieve all predictions.
			vector<LabelRBox> all_loc_preds;
			GetLocPredictionsR(loc_data, num_, num_priors_, regress_angle_, 
				regress_size_, &all_loc_preds);

			// Find matches between source rboxes and ground truth rboxes.
			vector<map<int, vector<float> > > all_match_overlaps;
			FindMatchesR(all_loc_preds, all_gt_rboxes, prior_rboxes, prior_variances,
				multirbox_loss_param_, &all_match_overlaps, &all_match_indices_);
			
			num_matches_ = 0;
			int num_negs = 0;
			// Sample hard negative (and positive) examples based on mining type.
			MineHardExamplesR(*bottom[1], all_loc_preds, all_gt_rboxes, prior_rboxes,
				prior_variances, all_match_overlaps, multirbox_loss_param_,
				&num_matches_, &num_negs, &all_match_indices_,
				&all_neg_indices_);
			
			/***************************************For Testing********************************************
			//LOG(INFO)<<all_gt_rboxes.size();
			ofstream fout("/home/all_gt_rboxes.txt",ios::trunc);
			for(int i=0;i<all_gt_rboxes.size();i++)
			{
				for(int j=0;j<all_gt_rboxes[i].size();j++)
					fout<< i <<" "<< j <<" "<<all_gt_rboxes[i][j].xcenter()<<" "<<all_gt_rboxes[i][j].ycenter()
					<<" "<<all_gt_rboxes[i][j].width()<<" "<<all_gt_rboxes[i][j].height()<<" "<<all_gt_rboxes[i][j].angle()<<"\n";
			}
			fout<<flush;fout.close();
			fout.open("/home/prior_rboxes.txt",ios::trunc);
			for(int i=0;i<prior_rboxes.size();i++)
			{
					fout<< i <<" "<<prior_rboxes[i].xcenter()<<" "<<prior_rboxes[i].ycenter()<<" "<<prior_rboxes[i].width()<<" "<<prior_rboxes[i].height()<<" "<<prior_rboxes[i].angle()<<"\n";
			}
			fout<<flush;fout.close();
			fout.open("/home/all_loc_preds.txt",ios::trunc);
			for(int i=0;i<all_loc_preds.size();i++)
			{
				for(int j=0;j<all_loc_preds[i][-1].size();j++)
					fout<<i<<" "<<j<<" "<<all_loc_preds[i][-1][j].xcenter()<<" "<<all_loc_preds[i][-1][j].ycenter()<<" "
					<<all_loc_preds[i][-1][j].width()<<" "<<all_loc_preds[i][-1][j].height()<<" "<<all_loc_preds[i][-1][j].angle()<<"\n";
			}
			fout<<flush;fout.close();
			fout.open("/home/all_match_indices.txt",ios::trunc);
			for(int i=0;i<all_match_indices_.size();i++)
			{
				for(int j=0;j<all_match_indices_[i][-1].size();j++)
					fout<<i<<" "<<j<<" "<<all_match_indices_[i][-1][j]<<" "<<all_match_overlaps[i][-1][j]<<"\n";
			}
			fout<<flush;fout.close();
			fout.open("/home/all_neg_indices.txt",ios::trunc);
			for(int i=0;i<all_neg_indices_.size();i++)
			{
				for(int j=0;j<all_neg_indices_[i].size();j++)
					fout<<i<<" "<<j<<" "<<all_neg_indices_[i][j]<<"\n";
			}
			fout<<flush;fout.close();
			LOG(INFO)<<"waiting key:";
			getchar();
			//LOG(FATAL)<<"Stop for debugging";
			/**************************************************************************************/
			int tmp = 2;
			if (regress_angle_) tmp ++;
			if (regress_size_) tmp += 2;
			if (num_matches_ >= 1) {
				// Form data to pass on to loc_loss_layer_.
				vector<int> loc_shape(2);
				loc_shape[0] = 1;
				loc_shape[1] = num_matches_ * tmp;
				loc_pred_.Reshape(loc_shape);
				loc_gt_.Reshape(loc_shape);
				Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
				Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
				EncodeLocPredictionR(all_loc_preds, all_gt_rboxes, all_match_indices_,
					prior_rboxes, prior_variances, multirbox_loss_param_,
					loc_pred_data, loc_gt_data);
				loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
				loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
			}
			else {
				loc_loss_.mutable_cpu_data()[0] = 0;
			}

			// Form data to pass on to conf_loss_layer_.
			if (do_neg_mining_) {
				num_conf_ = num_matches_ + num_negs;
			} else {
				num_conf_ = num_ * num_priors_;
			}
			//LOG(INFO)<<"num_matches_="<<num_matches_<<"   num_negs="<<num_negs;
			if (num_conf_ >= 1) {
				// Reshape the confidence data.
				vector<int> conf_shape;
				if (conf_loss_type_ == MultiRBoxLossParameter_ConfLossType_SOFTMAX) {
					conf_shape.push_back(num_conf_);
					conf_gt_.Reshape(conf_shape);
					conf_shape.push_back(num_classes_);
					conf_pred_.Reshape(conf_shape);
				} else if (conf_loss_type_ == MultiRBoxLossParameter_ConfLossType_LOGISTIC) {
					conf_shape.push_back(1);
					conf_shape.push_back(num_conf_);
					conf_shape.push_back(num_classes_);
					conf_gt_.Reshape(conf_shape);
					conf_pred_.Reshape(conf_shape);
				} else {
					LOG(FATAL) << "Unknown confidence loss type.";
				}
				if (!do_neg_mining_) {
					// Consider all scores.
					// Share data and diff with bottom[1].
					CHECK_EQ(conf_pred_.count(), bottom[1]->count());
					conf_pred_.ShareData(*(bottom[1]));
				}
				Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
				Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
				caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
				EncodeConfPredictionR(conf_data, num_, num_priors_, multirbox_loss_param_,
					all_match_indices_, all_neg_indices_, all_gt_rboxes,
					conf_pred_data, conf_gt_data);
				conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
				conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
			} else {
				conf_loss_.mutable_cpu_data()[0] = 0;
			}

			top[0]->mutable_cpu_data()[0] = 0;
			if (this->layer_param_.propagate_down(0)) {
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				top[0]->mutable_cpu_data()[0] +=
					loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
			}
			if (this->layer_param_.propagate_down(1)) {
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / normalizer;
			}
	}

	template <typename Dtype>
	void MultiRBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

			if (propagate_down[2]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to prior inputs.";
			}
			if (propagate_down[3]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to label inputs.";
			}
			
			int loc_num = 2;
			if (regress_size_) loc_num += 2;
			if (regress_angle_) loc_num ++;

			// Back propagate on location prediction.
			if (propagate_down[0]) {
				Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
				caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
				if (num_matches_ >= 1) {
					vector<bool> loc_propagate_down;
					// Only back propagate on prediction, not ground truth.
					loc_propagate_down.push_back(true);
					loc_propagate_down.push_back(false);
					loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
						loc_bottom_vec_);
					// Scale gradient.
					Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
						normalization_, num_, num_priors_, num_matches_);
					Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
					caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
					// Copy gradient back to bottom[0].
					const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
					int count = 0;
					for (int i = 0; i < num_; ++i) {
						for (map<int, vector<int> >::iterator it =
							all_match_indices_[i].begin();
							it != all_match_indices_[i].end(); ++it) {
								const int label = share_location_ ? 0 : it->first;
								const vector<int>& match_index = it->second;
								for (int j = 0; j < match_index.size(); ++j) {
									if (match_index[j] <= -1) {
										continue;
									}
									// Copy the diff to the right place.
									
									int start_idx = loc_classes_ * loc_num * j + label * loc_num;
									
									caffe_copy<Dtype>(loc_num, loc_pred_diff + count * loc_num,
										loc_bottom_diff + start_idx);
									++count;
									
									
								}
						}
						loc_bottom_diff += bottom[0]->offset(1);
					}
				}
			}

			// Back propagate on confidence prediction.
			if (propagate_down[1]) {
				Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
				caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
				if (num_conf_ >= 1) {
					vector<bool> conf_propagate_down;
					// Only back propagate on prediction, not ground truth.
					conf_propagate_down.push_back(true);
					conf_propagate_down.push_back(false);
					conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
						conf_bottom_vec_);
					// Scale gradient.
					Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
						normalization_, num_, num_priors_, num_matches_);
					Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
					caffe_scal(conf_pred_.count(), loss_weight,
						conf_pred_.mutable_cpu_diff());
					// Copy gradient back to bottom[1].
					const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
					if (do_neg_mining_) {
						int count = 0;
						for (int i = 0; i < num_; ++i) {
							// Copy matched (positive) rboxes scores' diff.
							const map<int, vector<int> >& match_indices = all_match_indices_[i];
							for (map<int, vector<int> >::const_iterator it =
								match_indices.begin(); it != match_indices.end(); ++it) {
									const vector<int>& match_index = it->second;
									//LOG(INFO) << "Match_index_size: "<<match_index.size();
									CHECK_EQ(match_index.size(), num_priors_);
									for (int j = 0; j < num_priors_; ++j) {
										if (match_index[j] <= -1) {
											continue;
										}
										// Copy the diff to the right place.
										caffe_copy<Dtype>(num_classes_,
											conf_pred_diff + count * num_classes_,
											conf_bottom_diff + j * num_classes_);
										++count;
										//LOG(INFO)<<"count2 = "<<count;
									}
							}
							//LOG(INFO)<<"POS: count="<<count;
							// Copy negative rboxes scores' diff.
							for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
								int j = all_neg_indices_[i][n];
								CHECK_LT(j, num_priors_);
								caffe_copy<Dtype>(num_classes_,
									conf_pred_diff + count * num_classes_,
									conf_bottom_diff + j * num_classes_);
								++count;
								//LOG(INFO)<<"count3 = "<<count;
							}
							//LOG(INFO)<<"NEG: count="<<count;
							conf_bottom_diff += bottom[1]->offset(1);
						}
					} else {
						// The diff is already computed and stored.
						bottom[1]->ShareDiff(conf_pred_);
					}
				}
			}

			// After backward, remove match statistics.
			all_match_indices_.clear();
			all_neg_indices_.clear();
	}

	INSTANTIATE_CLASS(MultiRBoxLossLayer);
	REGISTER_LAYER_CLASS(MultiRBoxLoss);

}  // namespace caffe
