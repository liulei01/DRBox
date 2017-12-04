#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/rdetection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void RDetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
  const Dtype* loc_data = bottom[0]->cpu_data();
	const Dtype* conf_data = bottom[1]->cpu_data();
	const Dtype* prior_data = bottom[2]->cpu_data();
	const int num = bottom[0]->num();
	
	gettimeofday(&t2, NULL);
	time0 += t2.tv_sec -t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	//LOG(INFO) << "time0 = " << time0;
	gettimeofday(&t1, NULL);
	
	// Retrieve all location predictions.
	vector<LabelRBox> all_loc_preds;
	GetLocPredictionsR(loc_data, num, num_priors_, num_loc_classes_,
		 share_location_, regress_angle_, regress_size_, &all_loc_preds);
		 
	gettimeofday(&t2, NULL);
	time1 += t2.tv_sec -t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	//LOG(INFO) << "time1 = " << time1;
	gettimeofday(&t1, NULL);

	// Retrieve all confidences.
	vector<map<int, vector<float> > > all_conf_scores;
	GetConfidenceScoresR(conf_data, num, num_priors_, num_classes_,
		&all_conf_scores);
		
	gettimeofday(&t2, NULL);
	time2 += t2.tv_sec -t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	//LOG(INFO) << "time2 = " << time2;
	gettimeofday(&t1, NULL);

	// Retrieve all prior rboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	vector<NormalizedRBox> prior_rboxes;
	vector<vector<float> > prior_variances;
	GetPriorRBoxes(prior_data, num_priors_, regress_angle_, regress_size_, 
	prior_width_, prior_height_, &prior_rboxes, &prior_variances);

	gettimeofday(&t2, NULL);
	time3 += t2.tv_sec -t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	//LOG(INFO) << "time3 = " << time3;
	gettimeofday(&t1, NULL);

	// Decode all loc predictions to rboxes.
	vector<LabelRBox> all_decode_rboxes;
	const bool clip_rbox = false;
	DecodeRBoxesAll(all_loc_preds, prior_rboxes, prior_variances, num,
		share_location_, num_loc_classes_, background_label_id_,
		code_type_, variance_encoded_in_target_, clip_rbox,
		regress_size_, regress_angle_, &all_decode_rboxes);
	
	gettimeofday(&t2, NULL);
	time4 += t2.tv_sec -t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	//LOG(INFO) << "time4 = " << time4;
	
	int num_kept = 0;
	vector<map<int, vector<int> > > all_indices;
	for (int i = 0; i < num; ++i)
	{
		const LabelRBox& decode_rboxes = all_decode_rboxes[i];
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		map<int, vector<int> > indices;
		int num_det = 0;
		const vector<float>& backscores = conf_scores.find(background_label_id_)->second;
		for (int c = 0; c < num_classes_; ++c)
		{
			if (c == background_label_id_)
			{
				// Ignore background class.
				continue;
			}
			if (conf_scores.find(c) == conf_scores.end())
			{
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find confidence predictions for label " << c;
			}
			const vector<float>& scores = conf_scores.find(c)->second;
			int label = share_location_ ? -1 : c;
			if (decode_rboxes.find(label) == decode_rboxes.end()) 
			{
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label " << label;
				continue;
			}
			const vector<NormalizedRBox>& rboxes = decode_rboxes.find(label)->second;
			ApplyNMSFastR(rboxes, scores, confidence_threshold_, nms_threshold_, eta_,
				top_k_, &(indices[c]));
			num_det += indices[c].size();
		}
		if (keep_top_k_ > -1 && num_det > keep_top_k_)
		{
			vector<pair<float, pair<int, int> > > score_index_pairs;
			for (map<int, vector<int> >::iterator it = indices.begin();
				it != indices.end(); ++it)
			{
					int label = it->first;
					const vector<int>& label_indices = it->second;
					if (conf_scores.find(label) == conf_scores.end())
					{
						// Something bad happened for current label.
						LOG(FATAL) << "Could not find location predictions for " << label;
						continue;
					}
					const vector<float>& scores = conf_scores.find(label)->second;
					for (int j = 0; j < label_indices.size(); ++j) 
					{
						int idx = label_indices[j];
						CHECK_LT(idx, scores.size());
						score_index_pairs.push_back(std::make_pair(
							scores[idx], std::make_pair(label, idx)));
					}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
				SortScorePairDescend<pair<int, int> >);
			score_index_pairs.resize(keep_top_k_);
			// Store the new indices.
			map<int, vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j)
			{
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += keep_top_k_;
		} 
		else 
		{
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(8);
	Dtype* top_data;
	if (num_kept == 0)
	{
		//LOG(INFO) << "Couldn't find any detections";
		top[0]->Reshape(top_shape);
		top_data = top[0]->mutable_cpu_data();
		caffe_set<Dtype>(top[0]->count(), -1, top_data);
		// Generate fake results per image.
		
		//top_data[0]=-1;
	}
	else
	{
		top[0]->Reshape(top_shape);
		top_data = top[0]->mutable_cpu_data();
		//top_data[0] = 0;
	}

	int count = 0;
	boost::filesystem::path output_directory(output_directory_);
	for (int i = 0; i < num; ++i)
	{
		//OG(INFO) << "Num=" <<num;
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		const vector<float>& backscores = conf_scores.find(background_label_id_)->second;
		const LabelRBox& decode_rboxes = all_decode_rboxes[i];
		if (need_save_) ++name_count_;
		boost::filesystem::path file(output_name_prefix_ + boost::lexical_cast<string>(name_count_) + ".txt");
		boost::filesystem::path out_file_name = output_directory / file;
		std::ofstream outfile;
		if (need_save_) 
		{
			outfile.open(out_file_name.string().c_str(),std::ofstream::out);
			//LOG(INFO) << out_file_name.string().c_str(),std::ofstream::out;
		}
		for (map<int, vector<int> >::iterator it = all_indices[i].begin();
			it != all_indices[i].end(); ++it)
		{
				int label = it->first;
				if (conf_scores.find(label) == conf_scores.end())
				{
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find confidence predictions for " << label;
					continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				int loc_label = share_location_ ? -1 : label;
				if (decode_rboxes.find(loc_label) == decode_rboxes.end())
				{
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find location predictions for " << loc_label;
					continue;
				}
				const vector<NormalizedRBox>& rboxes =
					decode_rboxes.find(loc_label)->second;
				vector<int>& indices = it->second;
				for (int j = 0; j < indices.size(); ++j)
				{
					int idx = indices[j];
					top_data[count * 8] = i;
					top_data[count * 8 + 1] = label;
					top_data[count * 8 + 2] = scores[idx];
					const NormalizedRBox& rbox = rboxes[idx];
					top_data[count * 8 + 3] = rbox.xcenter();
					top_data[count * 8 + 4] = rbox.ycenter();
					top_data[count * 8 + 5] = rbox.angle();
					top_data[count * 8 + 6] = rbox.width();
					top_data[count * 8 + 7] = rbox.height();
					++count;
					if (need_save_)
					{
						outfile << rbox.xcenter()*300 << " "<< rbox.ycenter()*300 << " " 
							<< rbox.width()*300 << " " << rbox.height()*300 << " "
							" " << label << " " << rbox.angle() << " " << scores[idx] << std::endl;
					}
				}
		}
		if (need_save_)
		{
			outfile.flush();
			outfile.close();
		}
	}
	
	
}

INSTANTIATE_LAYER_GPU_FUNCS(RDetectionOutputLayer);

}  // namespace caffe
