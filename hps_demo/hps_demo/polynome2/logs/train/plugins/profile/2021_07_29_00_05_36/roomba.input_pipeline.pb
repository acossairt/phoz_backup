	`???f??@`???f??@!`???f??@	J#??θ??J#??θ??!J#??θ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`???f??@˟o???A???????@Y??????*v??/ˈ@)      @=2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?ѓ2)??!@?e??R@)P?}:3??1??S???Q@:Preprocessing2U
Iterator::Model::ParallelMapV2??ɩ?a??! p??"@)??ɩ?a??1 p??"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat֫???$??!???hu?@)R?8?ߡ??1?|??RA@:Preprocessing2F
Iterator::Model??R?h??!J-????+@)z??Q???1Sz?!??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?k*??!	1IB?I@)?k*??1	1IB?I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?D??f???!W??̀U@)+???????1]p??? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg?ba????!g?w?$??)g?ba????1g?w?$??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????1J??!?`???R@)\?J?p?1LB}?>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9J#??θ??In4s??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	˟o???˟o???!˟o???      ??!       "      ??!       *      ??!       2	???????@???????@!???????@:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JCPU_ONLYYJ#??θ??b qn4s??X@