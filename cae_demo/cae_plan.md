你的上下文推理能力可以抽象为存算一体的机器。
我们现在需要基于这个能力，去完成一个全新的任务：测试时优化(testing time optimaztion)。
通过多轮对话，我们需要模拟：
    1. 前向
    2. 算loss
    3. 后向
    4. 参数更新
    5. 重复上述步骤，直到 loss 收敛。

为了更明确地说明如何模拟，我们做如下概念映射：
    1. 模型  <=> 神经编译器
    2. 张量  <=> 语义片段
        2.1 输入张量  <=> 源代码
        2.2 参数张量  <=> 语义映射
        2.3 输出张量  <=> 目标语义（可具象为代码）
    3. 前向计算  <=> 目标语义生成
    4. loss  <=> 神经编译器评估指标
    5. 后向计算  <=> 评估指标对神经编译器的优化指导
        5.1 梯度   <=> 语义修改指导
    6. 参数更新  <=> 语义映射更新
    7. 收敛  <=> 评估指标收敛

接下来，我们具体说明如何基于这个能力，去模拟 CAE (Code AutoEncoder) 的前向、后向、参数更新。

###     1. 模型  <=> 神经编译器
CAE

###     2. 张量  <=> 语义片段

语义片段 := list[
    Oneof
    | $SourceCode str
    | $SemanticMapping list[tuple[$source_code_fragment, $target_code_fragment str]
    | $TargetCode str
]

1. 输入张量可以来自于 viba 项目自身代码。
2. 参数张量可以来自于 cae_demo/weigts 下的所有语义映射。
    2.1 cae_demo/weigts目录内容示例:
    - cae_demo/weigts/semantic_mapping_name0/source.source_code_ext
    - cae_demo/weigts/semantic_mapping_name0/target.target_code_ext
    - cae_demo/weigts/semantic_mapping_name1/source.source_code_ext
    - cae_demo/weigts/semantic_mapping_name1/target.target_code_ext
    - cae_demo/weigts/semantic_mapping_name2/source.source_code_ext
    - cae_demo/weigts/semantic_mapping_name2/target.target_code_ext

###    3. 前向计算  <=> 目标语义生成
CAE的前向计算分成两步：
1. 源代码片段 X =>viba 语义片段，这里需要让大模型来生成。
    1.1 加载 cae_demo/weigts/ 下所有的映射关系到大模型上下文(需要用到 repomix)。
    1.2 通过cae_intent_prompt来引导大模型生成viba语义片段 I。（cae_intent_prompt当前没有想好，LLMs 自己根据本文思考一个）
    1.2 通过@viba/validate_cae_intent.py 来验证 I 是否合法
2. viba 语义片段=>目标代码
    2.1 通过@viba/get_truncated_intents.py 来将 I 做转换，得到不同压缩层级的 viba 语义片段 (itent_base, truncated_intents)。
    2.2 对于truncated_intents 中的每个 truncated_intent，通过cae_target_prompt来引导大模型基于 truncated_intent 生成代码 target_code。（cae_target_prompt当前没有想好，LLMs 自己根据本文思考一个）
    2.3 收集 target_code得到 target_code_list，显然 len(truncated_intents) == len(target_code_list)。

###     4. loss  <=> 神经编译器评估指标

调用@viba/get_cae_loss.py 获得当下的评估指标 loss。

###     5. 后向计算  <=> 评估指标对神经编译器的优化指导
内省@viba/get_cae_loss.py，看评估指标的各个分量，决定接下来的语义修改方向。

###     6. 参数更新  <=> 语义映射更新
更新 cae_demo/weigts/下所有语义映射

###     7. 收敛  <=> 评估指标收敛
正式版肯定是评估指标收敛，但是当前是测试版本，只需要 10 次迭代，我们先看看 loss 是否收敛。

### 其他内容
大模型需要记录每次迭代后 loss 的分数，还有语义映射的 diff，统一收集到 cae_demo/cae_train.log文件里。