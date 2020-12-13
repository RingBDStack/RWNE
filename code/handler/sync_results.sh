#!/usr/bin/env bash
# synchronize results on severs
# remote backup
results_dir='/home/heyu/work/result/NRL'
if [ ! -x "$results_dir" ];then 
	mkdir $results_dir
else
	echo -n "results already exist, whether to overwrite (or skiped)? (y/n) -> "
	read overwrite
	if [ $overwrite = y ] || [ $overwrite = Y ];then
		rm -r $results_dir
		mkdir $results_dir
	fi
fi
paths=()
while IFS='' read -r line || [[ -n "$line" ]]; do
	paths+=($line)
done < paths_list.txt
psw=()
while IFS='' read -r line || [[ -n "$line" ]]; do
	psw+=($line)
done < psw.txt
n=${#paths[*]}
for ((i=0;i<n;i++))
do
	p=${paths[$i]}
	name=${p##*/}
	if [ -d "$results_dir/$name" ]; then
		echo "duplicate name: $name,  skiped!"
#		echo "Copying from $name to ${name}_"
#		mv $results_dir/$name $results_dir/${name}_
	else


# parameters
mkdir $results_dir/$name
batch_size_list=(1 2 4 8 16 32 64 128)
walk_length_list=(1 2 4 8 16 32 64 128)
for ((p1=0;p1<${#batch_size_list[*]};p1++))
do
batch_size=batch_size_${batch_size_list[$p1]}
for ((p2=0;p2<${#walk_length_list[*]};p2++))
do
walk_length=walk_length_${walk_length_list[$p2]}


sshpass -p ${psw[$i]} rsync -avmr --delete \
--include="tmp/classify/best_ckpt.info" \
--include="tmp/classify/classify.info" \
-f 'hide,! */' \
$p/run/${batch_size}_${walk_length} $results_dir/$name

cp -r $results_dir/${name}/${batch_size}_${walk_length}/tmp/classify/* $results_dir/${name}/${batch_size}_${walk_length}/
rm -r $results_dir/${name}/${batch_size}_${walk_length}/tmp
#	cp -r $results_dir/${name}/log/* $results_dir/${name}/
#	rm -r $results_dir/${name}/log


done
done


    fi
done

