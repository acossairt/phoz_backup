	&??:Z@&??:Z@!&??:Z@	?UGD*2???UGD*2??!?UGD*2??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&??:Z@7???????Ao?;2V?Y@YK\Ǹ?b??*	??S?儧@2U
Iterator::Model::ParallelMapV2?D?$]s??!/{??NG@)?D?$]s??1/{??NG@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?S?4????!;H?>??G@)U??????1-?m+?F@:Preprocessing2F
Iterator::Model]¡?xx??!CV??fI@)???(??1??\#F?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?<0???!?Am0?n??)?<0???1?Am0?n??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,J	??z??!?ᱤ?r??)?1!撪??1?????}??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??r?S???!???(?H@)9?Վ???10?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?@׾?~?!Bpi?????)?@׾?~?1Bpi?????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??L?D???!?At???G@)??zm?1?5???4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?UGD*2??I???V7?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7???????7???????!7???????      ??!       "      ??!       *      ??!       2	o?;2V?Y@o?;2V?Y@!o?;2V?Y@:      ??!       B      ??!       J	K\Ǹ?b??K\Ǹ?b??!K\Ǹ?b??R      ??!       Z	K\Ǹ?b??K\Ǹ?b??!K\Ǹ?b??b      ??!       JCPU_ONLYY?UGD*2??b q???V7?X@