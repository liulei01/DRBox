root_dir=/home # This is your Caffe root directory 
cd $root_dir

redo=1
data_root_dir="$root_dir/data/Airplane/train_data"
dataset_name="Airplane"
mapfile=""
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
label_type="txt"

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in trainval
do
  python $root_dir/scripts/create_annoset_r.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --backend=$db --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
