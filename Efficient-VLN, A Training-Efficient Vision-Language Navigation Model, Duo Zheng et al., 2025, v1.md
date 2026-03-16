Efficient-VLN: A Training-Efficient Vision-Language Navigation Model
Duo Zheng
Shijia Huang
Yanyang Li
Liwei Wang♯
The Chinese University of Hong Kong
LN
Abstract
## arXiv:2512.10310v1  [cs.CV]  11 Dec 2025
Multimodal large language models (MLLMs) have shown
promising potential in Vision-Language Navigation (VLN).
However, their practical development is severely hindered
by the substantial training overhead. We recognize two key
issues that contribute to the overhead: (1) the quadratic
computational burden from processing long-horizon histor-
ical observations as massive sequences of tokens, and (2)
the exploration-efficiency trade-off in DAgger, i.e., a data
aggregation process of collecting agent-explored trajecto-
ries. While more exploration yields effective error-recovery
trajectories for handling test-time distribution shifts, it
comes at the cost of longer trajectory lengths for both train-
ing and inference. To address these challenges, we pro-
pose Efficient-VLN, a training-efficient VLN model. Specif-
ically, to mitigate the token processing burden, we design
two efficient memory mechanisms: a progressive memory
that dynamically allocates more tokens to recent observa-
tions, and a learnable recursive memory that utilizes the
key-value cache of learnable tokens as the memory state.
Moreover, we introduce a dynamic mixed policy to balance
the exploration-efficiency trade-off. Extensive experiments
show that Efficient-VLN achieves state-of-the-art perfor-
mance on R2R-CE (64.2% SR) and RxR-CE (67.0% SR).
Critically, our model consumes merely 282 H800 GPU
hours, demonstrating a dramatic reduction in training over-
head compared to state-of-the-art methods.
Figure 1.
Efficient-VLN sets a new state-of-the-art in Vision-
Language Navigation in Continuous Environments using only
RGB inputs. We compare the Success Rate (SR) on the R2R-
CE benchmark (y-axis) against training overhead in GPU hours
(x-axis, log scale). The size of each bubble represents the rela-
tive model size. Our method achieves 64.2% R2R SR and 67.0%
RxR SR, surpassing previous state-of-the-art methods. Notably,
Efficient-VLN achieves this using only 282 H800 GPU hours, a
fraction of the compute required by competing methods.
Navigation (VLN) [2, 19, 21, 25, 29] has emerged as a
fundamental research problem, garnering extensive interest
[5, 7, 10, 20, 41–43, 46, 49] from the community.
A prominent line of recent work [10, 23, 35, 41, 43, 49]
tackles this problem by extending multimodal large lan-
guage models (MLLMs) to this field, via modeling the
historical observations as video inputs.
Benefiting from
the inherent world knowledge and generalizability, this
paradigm has demonstrated superior performance and sig-
nificant promise. Nevertheless, training these MLLM-based
models presents a substantial challenge owing to their im-
mense training overhead. As shown in Figure 1, the training
of StreamVLN [35] and NavFoM [42] required 1,500 A100
and 4,032 H100 GPU hours, respectively. This substan-
tial training overhead hinders researchers, especially those
with limited computational budgets, from training advanced
VLN models.
1. Introduction
Building autonomous robots that can perceive, understand,
and intelligently interact with the physical world represents
a long-standing goal in artificial intelligence. A critical as-
pect is the ability to navigate following human’s natural lan-
guage instructions, playing a pivotal role in diverse practical
scenarios, such as fulfilling domestic tasks, navigating com-
plex warehouses, and rescuing in hazardous environments.
To address this challenge, the task of Vision-Language
♯corresponding author
1
![image](images/pdf_image_1_1.jpeg)

![image](images/pdf_image_1_2.png)

# In this work, we focus on two key issues contributing
to this challenge. First, VLN models involve processing
extensive historical observations. These observations are
tokenized into a sequence of tokens whose length is often
excessively large, due to either accumulating all temporal
steps [41] or using dense per-frame visual tokens [10, 35].
As a result, this massive token count imposes a quadratic
computational burden and dominates the overall training
overhead.
Second, to enhance error recovery, the previ-
ous work [19, 35] employs DAgger, which collects off-
policy trajectories based on a mixed policy. Specifically,
the mixed policy selects the oracle or learned policy based
on a fixed ratio, to generate the executed actions. A high ra-
tio of learned policy-generated actions drives the deviation
from the correct path, and this benefits learning to handle
the test-time distribution shift. However, it also results in
substantially longer exploration trajectories and even task
failure, thus raising the training overhead.
To address the above issues, we propose a training-
efficient VLN model, namely Efficient-VLN. First, to tackle
the computational overhead of processing massive mem-
ory tokens, we introduce two novel efficient memory rep-
resentations. (1) A progressive memory representation is
constructed via a progressive spatial compression, which is
inspired by the human forgetting mechanism (i.e., dimin-
ishing of memories over time). Specifically, it applies a
low compression ratio to visual tokens from recent frames,
while gradually increasing this ratio for temporally distant
ones. (2) A learnable recursive memory utilizes the KV
cache of a set of learnable tokens to serve as the memory
state. By propagating the memory state across consecutive
steps, the memory is recursively updated while maintaining
a fixed size. To enrich these representations, we integrate
3D geometry information into 2D appearance visual fea-
tures. This 3D geometry information is extracted by a 3D
geometry encoder (e.g., StreamVGGT [53]) from raw RGB
videos, without requiring explicit depth sensors or maps.
Second, we introduce a dynamic mixed policy to better bal-
ance the exploration-efficiency trade-off in DAgger. Specif-
ically, this policy starts with the learned policy for accumu-
lating the compounding errors, and then progressively shifts
towards the oracle policy to ensure eventual task comple-
tion. This design enables the collection of error-recovery
data without incurring excessively long trajectories, thus ac-
celerating the training process.
We evaluate our method on widely used VLN bench-
marks, including R2R-CE [19] and RxR-CE [21]. The pro-
posed model achieves state-of-the-art performance with a
64.2% SR on R2R-CE and 67.0% SR on RxR-CE, while
consuming merely 282 H800 GPU hours, which is far less
than competing methods (Figure 1). Our experiments reveal
several interesting findings:
• The progressive memory representation excels at the
long-horizon RxR-CE benchmark, and demonstrates con-
tinued improvement as the memory window increases.
• The recursive memory performs well on short trajecto-
ries (R2R-CE) but struggles on longer ones (RxR-CE),
demonstrating the challenge of maintaining long-term
memory for state-based models.
• By utilizing the dynamic mixed policy, we achieve higher
SR with a 56% reduction in exploration overhead.
2. Related Work
Vision-Language Navigation (VLN). VLN [2, 19, 21, 25,
29] represents a fundamental challenge in embodied AI,
with prior work [5–7, 12, 20, 32, 46] encompassing var-
ious techniques, such as multi-task learning [43, 46, 52],
specialized memory structures [7, 13, 32, 34], and data aug-
mentation [28, 31, 38]. Our recursive memory approach
is closely related to VLN
⟳
BERT [12], which pioneers the
use of a recursive state token to compress history. It em-
ploys a single state token that is continuously processed
by a BERT-style architecture. However, this poses signif-
icant challenges for gradient propagation in deep, multi-
layer MLLMs. We address this limitation by introducing
a novel recursive mechanism based on the KV cache. An-
other closely related work is CorrectNav [38], which gener-
ates error-recovery data with an LLM-based framework and
iteratively bootstraps the model. However, this approach’s
reliance on LLM queries incurs additional cost, whereas our
approach does not require external tools.
MLLMs for Navigation. Recent research [10, 23, 34, 35,
41, 43, 46, 49] in VLN has focused on adapting Multimodal
Large Language Models (MLLMs) for embodied naviga-
tion to leverage their inherent generalizability to unseen
environments. Several studies [23, 49] explore employing
proprietary MLLMs as navigation agents in a zero-shot set-
ting. More recent efforts [10, 34, 35, 41, 43, 46] have shifted
towards training specialized models that leverage advanced
MLLMs as the backbone. NaviLLM [46] and NaVid [41]
pioneer the development of trainable MLLM-based models
in embodied navigation. Subsequently, NaVILA [10] in-
troduces hierarchical agents that decouple high-level plan-
ning from low-level execution, while StreamVLM [35] en-
hances inference efficiency using a non-overlapping sliding
window mechanism. Despite the significant progress, these
methods are often constrained by substantial training over-
head. The primary objective of this work is to propose a
framework for training-efficient VLN models.
MLLMs for Spatial Intelligence. Understanding the 3D
world from video inputs is a cornerstone of spatial in-
telligence and has attracted widespread academic interest
[9, 37, 47, 48]. Some approaches [9, 48] enhance the per-
ception of 3D scenes by integrating explicit 3D information
2
into MLLMs. Another line of work [36, 47] draws inspira-
tion from the success of 3D geometry model [22, 30, 53],
and enhances spatial reasoning by integrating latent 3D to-
kens extracted from RGB frames into their visual represen-
tation, obviating the necessity for 3D sensor inputs. While
these studies [36, 47] primarily focus on static scene under-
standing, i.e., from offline videos of static scenes, our work
extends this paradigm to handle the real-time dynamic ob-
servations in the navigation process. A concurrent work,
JanusVLN [39], also explores adapting this paradigm to
VLN. Our approach further distinguishes itself by focus-
ing on efficient memory representations and data collection
strategies for error correction.
MLLM
text token 𝒘
visual token 𝒇𝒕
memory token 𝑚%&'
t
t-∆
t-2∆
t-3∆
t-4∆
t-5∆
t-6∆
Cache
𝐾𝑉"#$
MLLM
𝐾𝑉!
KV
3. Method
text token 𝒘
visual token 𝒇𝒕
memory token 𝒎𝒄𝒖𝒓
sentinel token 𝒎𝒄𝒖𝒓
3.1. Overview
Figure 2. Illustrations of two distinct memory paradigms designed
for MLLMs. (Top) The progressive memory representation allo-
cates tokens based on temporal recency. (Bottom) The recursive
memory representation maintains memory state by utilizing the
KV cache of sentinel tokens to preserve crucial information.
Task Formalization. In the Vision-Language Navigation
(VLN) task, a model is tasked with navigating to a target
location following a language instruction I. At each step
t, the model receives a new image vt as the current ob-
servation and predicts the next sequence of actions condi-
tioned on the instruction I and the full observation history
{v1, v2, · · · , vt}. Following the previous work [35, 43], the
model is asked to predict the next four actions from a dis-
crete action space comprising forward, left, right,
and stop. An episode terminates when the model produces
the stop action.
MLLM-based approaches [10, 35, 41] have predominantly
focused on processing RGB frame sequences and over-
looked the explicit modeling of geometry information for
3D scenes. To address this limitation, our work enriches the
visual representation by injecting 3D geometry information
extracted by a 3D geometry encoder into the conventional
2D visual features. A 3D geometry encoder typically esti-
mates geometric attributes (e.g., depth) directly from RGB
images, thereby eliminating the requirement for dedicated
depth sensors or pre-built maps.
Architecture.
Our model consists of three components:
a visual encoder, a 3D geometry encoder, and an MLLM
backbone. The workflow at step t involves the following
three steps. First, the current observation vt is encoded into
a geometry-enhanced visual representation ft, which fuses
conventional 2D visual features with 3D geometry features
from the 3D geometry encoder. Then, the textual embed-
dings of the instruction I, the current geometry-enhanced
visual representation ft, and the memory representation are
provided to the MLLM backbone to generate the action se-
quence. Last, the memory representation is updated with
this new visual representation (ft) to maintain crucial infor-
mation.
We first describe the geometry-enhanced visual rep-
resentation in Section 3.2, and then present two novel
training-efficient memory representations in Section 3.3.
Finally, we detail our training strategy and propose a dy-
namic mixed policy to enhance the DAgger algorithm in
Section 3.4.
2D Visual Encoding.
Given an RGB image vt at step
t, the model first employs the visual encoder to convert
each image vt ∈Rh×w×3 into a series of visual tokens
ˆvt ∈R⌊h
p⌋×⌊w
p ⌋×c, where p represents the patch size.
In this work, our model is built upon Qwen2.5-VL [4],
which further compresses the visual features by grouping
spatially adjacent 2 × 2 tokens into a single visual to-
ken. This process yields the final 2D visual representation
vt ∈R⌊h
2p⌋×⌊w
2p⌋×c.
Incorporating 3D Geometry into 2D Visual Represen-
tation. Then, we follow [47] to employ a 3D geometry
model to extract the geometry information from raw RGB
frame sequences. StreamVGGT is adopted in our work, as
it is designed for streaming processing, and thus capable of
handling dynamically updating observations. Despite this,
the growing frame count causes extensive memory costs.
To mitigate this problem, we evict the key-value states of
a random frame (excluding the reference frame) from the
cache if the number of cached frames exceeds the prede-
3.2. Geometry-Enhanced Visual Representation
Understanding the geometric structure of a 3D scene is
crucial for recalling previously observed landmarks and
achieving accurate spatial localization.
However, prior
3
fined limit StreamVGGT receives the current observation
vt, and produces 3D geometry latent tokens by reusing the
KV cache of past images, Subsequently, these 3D geom-
etry latent tokens are processed through a 2-layer MLP to
align them with the 2D visual representation, resulting in
the final 3D geometry representation gt ∈R⌊h
count of memory features is derived as:
∞
X
4i )S = KS
K
4 · S + K
16 · S + K
64 · S + · · · = K
1
4 (
3 .
i=0
This approach facilitates a more effective token budget allo-
cation. K is set 3, to ensure the total count for observation
history does not exceed that of a single additional image.
2p⌋×⌊w
2p⌋×c.
Finally, the 2D visual representation vt and the 3D geom-
etry representation gt are fused via element-wise addition,
yielding the geometry-enhanced visual representation, de-
noted as ft = vt + gt.
Recursive Memory Representation A conventional strat-
egy for mitigating token consumption is recursive memory,
where a fixed-size state is iteratively updated. However,
adapting this to MLLMs presents significant challenges.
The primary difficulty stems from propagating gradients ef-
fectively through deep, multi-layer architectures across ex-
tended steps.
We introduce a novel approach that enables recursive
memory modeling in MLLMs.
Our intuition is to em-
ploy the KV cache of learnable tokens as the memory state.
Specifically, the input prompt at time step t is structured as:
3.3. Training-Efficient Memory Representation
Existing approaches [35, 41, 43] leveraging advanced
MLLMs often impose a significant training overhead due to
processing a massive number of visual tokens. For example,
in each training step, NaVILA [10] processes 8 frames and
consumes up to 196 × 8 = 1568 tokens, while StreamVLN
[35] utilizes up to 196 × 16 = 3136 tokens for its visual
inputs. To this end, we propose a two-fold strategy. First,
we apply a general temporal sampling technique to reduce
frame-level redundancy. Second, building upon this sam-
pled history, we introduce two novel and efficient memory
representations: progressive memory representation and re-
cursive memory representation. These representations, il-
lustrated in Figure 2, are designed to further reduce token
consumption and accelerate the training process.
{ft}, {mpre}, {w}, {mcur}
ft and w denote the current visual representation and the in-
struction’s word embeddings, respectively. mpre serves as a
placeholder for history memory, while mcur represents the
learnable tokens used to compute the current memory state.
As depicted in the Figure 2, within each transformer block
of MLLM, the key-value (KV) states of the placeholder
mpre are replaced by the KV states of {mcur} computed
at the previous step t −∆. During the forward pass at step
t, the model processes this composite prompt. The learn-
able tokens {mcur} attend to both the current inputs ({ft},
{w}) and the past memory state (via {mpre}).This mecha-
nism updates the internal states of {mcur}, integrating in-
formation from the current step and the historical context.
The recursive process enables {mcur} to function as an
evolving memory state, propagating context forward across
successive steps. Furthermore, utilizing the KV cache for
learnable tokens, as opposed to output hidden states, mit-
igates the computational challenges associated with long-
range gradient propagation during backpropagation.
In this paper, we adopt the progressive memory repre-
sentation as the default configuration for our model. All
main experiments, including comparisons with state-of-the-
art methods, utilize this setup. We provide a detailed com-
parative analysis of the recursive memory representation in
the experimental analysis section in Section 4.4.
Temporal Sampling. Since adjacent frames often contain
visually correlated content, processing these frames gener-
ates an excessive number of visual tokens and introduces
significant computational redundancy. To reduce such re-
dundancy, we apply temporal sampling by uniformly select-
ing frames from the sequence at a stride ∆.
Progressive Memory Representation Inspired by hu-
man forgetting mechanisms, which allocate greater re-
sources to recent memories and progressively fewer to
older ones, we introduce a progressive memory represen-
tation. This approach simulates the forgetting process by
applying spatial compression to visual tokens of past ob-
servations.
Given the visual features of sampled frames
{· · · , ft−2∆, ft−∆, ft}, where fi ∈R⌊h
2p⌋×⌊w
2p⌋×c, the K
most recent feature maps are downsampled by a factor of
2 × 2. Subsequently, the next group of K feature maps is
downsampled by a 4×4 factor. This iterative procedure con-
tinues for subsequent groups, applying progressively larger
downsampling factors, until the feature maps’ dimensions
preclude further reduction. This approach is analogous to
storing recent history at high resolution and distant history
at increasingly lower resolutions. Assuming a single im-
age comprises S tokens, the upper bound for the total token
3.4. Training Strategy
We follow StreamVLN [35] to employ a two-stage train-
ing strategy. The primary objective of the first stage is to
equip the MLLM with fundamental navigation capabilities.
Therefore, the model is trained on a mixed dataset of R2R-
CE [19] and RxR-CE [21]. Different from StreamVLN, we
4
Algorithm 1 DAgger with Dynamic Mixed Policy
low the shortest path guided by the oracle policy, and is thus
ineffective for enhancing error recovery.
To resolve this issue, we propose using a dynamic mixed
policy to improve DAgger, as illustrated in Algorithm 1.
To be specific, the probability β is dynamically adjusted as
navigation progresses, determined by the following equa-
tion: β = 1 −α
t
Require: Current policy πθ, Oracle policy π∗, Dataset D
Require: Set of environments E, Decay rate α, Num tra-
jectories N
1: Dnew ←∅
2: for i = 1 →N do
▷Collect N new trajectories
T , where α denotes a decay rate, t is the
current time step, and T represents the total number of steps
in the ground-truth path. β increases from 0 towards 1 as the
navigation progresses, facilitating a gradual transition from
the learned policy towards the oracle policy. Consequently,
this design enables collecting exploratory trajectories with-
out incurring excessive exploration lengths.
3:
Sample env e ∼E with state s0 and path length T
4:
t ←0
5:
while st is not terminal do
6:
βt ←1 −αt/T
▷Calculate dynamic oracle
probability
7:
at ∼βtπ∗(·|st) + (1 −βt)πθ(·|st)
▷Sample
action to execute
8:
a∗
t ←π∗(st) ▷Get expert label for visited state
Training Acceleration via Sequence Packing.
During
training, a common step-by-step approach, which performs
a separate forward pass per action step, leads to significant
training overhead due to inefficient GPU utilization. To ad-
dress this, we employ a sequence packing strategy that con-
catenates token sequences from multiple consecutive steps
into a single flattened sequence. A block sparse attention
mask1 is then employed to confine attention computations
to their respective steps. Furthermore, this concatenation
strategy is crucial for our recursive memory representation,
as it enables gradient backpropagation across steps.
By
adopting this method, we double the number of steps pro-
cessed per backward pass (from 8 to 16), which reduces the
total training overhead by 41.2%.
9:
Dnew ←Dnew ∪{(st, a∗
t )}
10:
st+1 ←EnvironmentStep(st, at)
11:
t ←t + 1
12:
end while
13: end for
14: Dagg ←D ∪Dnew
▷Aggregate datasets
15: πθnew ←Train(Dagg)
▷Train new policy
16: return πθnew
excluded the EnvDrop [28] dataset, as our preliminary stud-
ies indicate that extending the training steps on the mixed
dataset yields greater performance benefits than incorporat-
ing the EnvDrop data.
In the second stage, we aim to enhance the model’s abil-
ity to correct errors, generalize to unseen environments,
and follow instructions. This is achieved by augmenting
the training corpus with additional data. First, we utilize
ScaleVLN-150K [31] to improve generalization. Then, to
enhance multimodal understanding, we incorporate multi-
modal question-answering datasets, including ScanQA [3],
SQA3D [24], and a subset of LLaVA-Video-178K [45]. Fi-
nally, to specifically enhance error recovery, we introduce a
dynamic mixed policy within the DAgger algorithm to col-
lect valuable error-recovery data.
4. Experiments
4.1. Implementation and Training Details
Our model is built upon Qwen2.5-VL-3B [4] and utilizes
StreamVGGT-1B [53] as the 3D geometry encoder. The
progressive memory representation is used by default for
training and evaluation unless specified otherwise. The tem-
poral sampling stride ∆is set to 4, and the sliding window
size N is set to 12. For both training stages, we employ the
Adam optimizer with a batch size of 128 and a warmup ratio
of 0.03. The learning rate is gradually increased to 1e-5 dur-
ing warmup before linearly decaying to 0. During training,
we freeze the MLLM’s visual encoder, the 3D geometry
encoder, and the multimodal connector, while leaving the
MLLM backbone trainable. All experiments are conducted
on 8 H800 80G GPUs.
Both the data collection and evaluation processes are
conducted within the Habitat simulator. For the navigation
task, we adopt the configuration from NaVILA [10], set-
ting the visual observation resolution to 512 × 512 and the
Horizontal Field of View (HFOV) to 90◦. Each navigation
episode terminates once the agent’s step count reaches the
maximum of 500.
Improving DAgger with a Dynamic Mixed Policy. Pre-
vious work [19, 41, 43] employs the Dataset Aggregation
(DAgger) [27] algorithm to enhance error recovery. DAg-
ger collects off-policy trajectories by executing a mixed pol-
icy, which selects between the learned and oracle policy us-
ing a fixed probability β for the oracle policy. However,
the choice of β introduces an exploration-efficiency trade-
off. Specifically, a lower value of β drives a greater devia-
tion from the correct path. This deviation is beneficial for
learning to handle the test-time distribution shift and recov-
ering from errors; however, it also results in substantially
longer exploration trajectories, thereby increasing the train-
ing overhead. In contrast, a higher value of β tends to fol-
1We adopt the Flex-FlashAttention implemented by [40].
5
Method
Observation Encoder
R2R-CE Val-Unseen
RxR-CE Val-Unseen
Pano.
Odo.
Depth
S.RGB
NE↓
OS↑
SR↑
SPL↑
NE↓
SR↑
SPL↑
nDTW↑
HPN+DN∗[20]
✓
✓
✓
6.31
40.0
36.0
34.0
-
-
-
-
CMA∗[13]
✓
✓
✓
6.20
52.0
41.0
36.0
8.76
26.5
22.1
47.0
⟳
BERT∗[13]
✓
✓
✓
5.74
53.0
44.0
39.0
8.98
27.0
22.6
46.7
VLN
Sim2Sim∗[18]
✓
✓
✓
6.07
52.0
43.0
36.0
-
-
-
-
GridMM∗[32]
✓
✓
✓
5.11
61.0
49.0
41.0
-
-
-
-
ETPNav∗[1]
✓
✓
✓
4.71
65.0
57.0
49.0
5.64
54.7
44.8
61.9
ScaleVLN∗[31]
✓
✓
✓
4.80
–
55.0
51.0
-
-
-
-
InstructNav [23]
-
-
-
-
6.89
–
31.0
24.0
-
-
-
-
R2R-CMTP [5]
✓
✓
✓
7.90
38.0
26.4
22.7
-
-
-
-
LAW [26]
✓
✓
✓
6.83
44.0
35.0
31.0
10.90
8.0
8.0
38.0
CM2 [11]
✓
✓
✓
7.02
41.5
34.3
27.6
-
-
-
-
WS-MGMap [6]
✓
✓
✓
6.28
47.6
38.9
34.3
-
-
-
-
ETPNav + FF [33]
✓
✓
✓
5.95
55.8
44.9
30.4
8.79
25.5
18.1
-
Dynam3D [34]
✓
✓
✓
5.34
62.1
52.9
45.7
-
-
-
-
NaVid [41]
✓
5.47
49.1
37.4
35.9
-
-
-
-
NaVILA [10]
✓
5.37
57.6
49.7
45.5
-
-
-
-
StreamVLN [35]
✓
5.43
62.5
52.8
47.2
6.72
48.6
42.5
60.2
Efficient-VLN
✓
4.36
69.0
60.8
53.7
4.40
63.5
52.1
66.8
NaVILA† [10]
✓
5.22
62.5
54.0
49.0
6.77
49.3
44.0
58.8
UniNaVid† [43]
✓
5.58
53.3
47.0
42.7
6.24
48.7
40.9
-
StreamVLN† [35]
✓
4.98
64.2
56.9
51.9
6.22
52.9
46.0
61.9
NavFoM† [42]
✓
5.01
64.9
56.2
51.2
5.51
57.4
49.4
60.2
Efficient-VLN†
✓
4.18
73.7
64.2
55.9
3.88
67.0
54.3
68.4
Table 1. Comparison with state-of-the-art methods on R2R-CE and RxR-CE Val-Unseen split. The “Observation Encoder” inputs include
panoramic (Pano.), odometry (Odo.), depth image (Depth), and single RGB image (S.RGB). ∗denotes methods utilizing the waypoint
predictor from [13]. † denotes methods trained with additional data external to Matterport 3D.
Rate (SR), the percentage of the episodes where the agent
stops within a predefined distance threshold of the target; 3)
Oracle Success Rate (OSR), the SR given an oracle stop-
ping policy; 4) Success rate weighted by the Path Length
(SPL), the SR weighted by the ratio of the ground truth
length to the agent’s path length; 5) normalized Dynamic
Time Warping (nDTW) [17], which applies time warping
to measure the alignment between the agent’s trajectory and
ground truth path.
Training cost (hours)
# Samples
# Trajectories
UniNavid
1400 (H800)
3.6M
–
NaVILA
576 (A100)
1.5M
181K
StreamVLN
1500 (A100)
–
990K
NavFoM
4032 (H100)
12.7M
–
Efficient-VLN
282 (H800)
3.7M
196K
Table 2. Comparison of training costs and data volumes against
other methods. #Samples denotes the count of (state, action) pairs,
and #Trajectories is the number of navigation trajectories.
4.3. Quantitative Results
Table 1 compares our model with state-of-the-art methods
on R2R-CE and RxR-CE Val-Unseen benchmarks. We first
evaluate our method trained without external navigation
data. Efficient-VLN achieves an SR of 60.8% on R2R-CE
and 63.5% on RxR-CE, outperforming the previous state-
of-the-art StreamVLN by 8.0% and 14.9%, respectively.
When further incorporating the ScaleVLN-150K dataset,
our model sets a new state-of-the-art, achieving an SR of
64.2 on R2R-CE and an SR of 67.0 on RxR-CE.
Table 2 compares the training costs and data volumes
with those of competing methods.
Notably, our method
requires only 282 H800 GPU hours, a significantly lower
training cost than prior state-of-the-art models, highlighting
the training efficiency of our proposed approach.
4.2. Evaluation Setup
Benchmarks.
We evaluate our method on two widely
used VLN benchmarks, i.e., R2R-CE [19] and RxR-CE
[21], both based on the Habitat simulator. R2R-CE com-
prises 4,475 trajectories, split into 3,862 for training and
613 for the val-unseen split, with each trajectory accom-
panied by three natural language instructions. RxR-CE in-
cludes 14,005 trajectories, consisting of 10,336 for training
and 3,669 for the val-unseen split.
Metrics. We employ the following metrics: 1) Navigation
Error (NE), which measures the average distance between
the agent’s final location and the destination; 2) Success
6
R2R-CE Val-Unseen
RxR-CE Val-Unseen
#
NE↓
OS↑
SR↑
SPL↑
#Token
NE↓
SR↑
SPL↑
nTDW↑
#Token
1
Spatial Compression (all frames)
4.79
66.9
55.9
48.8
499
5.13
58.6
47.6
64.1
606
2
Recursive Memory (64 tokens)
4.65
66.6
56.1
49.4
586
5.46
54.7
45.3
63.8
677
3
Prog. Spatial Compression (3 frames)
4.47
67.5
58.5
52.3
661
4.71
61.9
50.6
65.1
745
4
Prog. Spatial Compression (6 frames)
4.23
69.4
61.3
55.1
692
4.44
62.4
51.3
66.5
780
5
Prog. Spatial Compression (12 frames)
4.36
69.0
60.8
53.7
701
4.40
63.5
52.1
66.8
785
Table 3. Comparison of different memory modeling strategies. “#Token” denotes the average token count per forward pass during naviga-
tion. The ScaleVLN dataset is excluded from these experiments.
R2R-CE Val-Unseen
RxR-CE Val-Unseen
#
NE↓
OS↑
SR↑
SPL↑
#Train
Step
#Infer
Step
NE↓
SR↑
SPL↑
nDTW↑
#Train
Step
#Infer
Step
1
Baseline (w/o DAgger)
6.41
54.5
45.9
41.9
–
83
6.51
49.8
41.5
59.4
–
137
2
Constant Ratio (β = 0.75 )
5.77
57.1
50.7
46.7
66
84
5.71
54.1
46.4
64.2
98
126
3
Constant Ratio (β = 0.5)
4.95
61.5
54.8
49.5
78
94
4.79
60.7
50.7
69.0
112
142
4
Constant Ratio (β = 0.25)
3.97
69.6
59.5
52.1
128
146
4.01
62.6
51.6
67.7
160
186
5
Dynamic Ratio (α = 0.5)
4.36
69.0
60.8
53.7
82
100
4.40
63.5
52.1
66.8
121
154
Table 4. Effect of DAgger algorithm. #Train Step represents the average length of collected trajectories. #Infer Step is the average length
of predicted trajectories during inference. The ScaleVLN dataset is excluded from the experiments.
4.4. Analysis
metrics, while only increasing the average token count by
∼30. Further increasing frames to 12 (row 5) continues to
improve performance on RxR-CE, though it incurs a minor
performance drop on R2R-CE. These findings suggest that
the progressive memory is particularly effective for long-
horizon navigation, as evidenced by the consistent gains on
the longer-trajectory RxR-CE dataset.
### Effect of Memory Representation. We present the com-
parison of different memory representations in Table 3.
Specifically, we investigate the following variants. 1) “Re-
cursive Memory”: We use the KV cache of 64 learnable
tokens as the recursive memory representation. 2) “Spatial
Compression”: Following the prior work [41], we compress
each image into four tokens. 3) “Prog. Spatial Compres-
sion”: we retain the n most recent images (n ∈{3, 6, 12})
and employ the progressive spatial compression to encode
them into the memory. For all variants, video frames are
sampled with a stride of four. In addition to standard nav-
igation metrics, we also report the average token count per
forward pass (#Token) during navigation.
“Recursive Memory” (row 2) obtains 56.1 SR on R2R-
CE and 54.7 SR on RxR-CE, which is highly competitive
with the previous state-of-the-art, StreamVLN (52.8 SR on
R2R-CE, 48.6 on RxR-CE). Comparing rows 1 and 2, we
observe that while “Recursive Memory” outperforms on
R2R-CE, it underperforms in RxR-CE. Given that RxR-CE
features longer trajectories, this indicates that the “Recur-
sive Memory” struggles to retain essential information over
long sequences. Comparing rows 1 and 3, “Prog. Spatial
Compression (3 frames)” significantly surpasses “Spatial
Compression” by 2.6 SR on R2R-CE and 3.0 SR on RxR-
CE. This highlights the importance of recent visual history,
suggesting that aggressively compressing this memory (as
in row 1) hinders performance. Increasing frames from 3 to
6 (rows 3 vs. 4) yields significant improvements across all
Effect of DAgger Algorithm We further investigate the ef-
fect of the DAgger algorithm in Table 4. “Baseline” denotes
the model trained without DAgger data. “Constant ratio”
applies a fixed mixing ratio β for selecting the oracle pol-
icy, while “Dynamic ratio” employs a dynamic mixing ra-
tio. Comparing rows 1 and 2, incorporating the DAgger data
yields a significant improvement, boosting SR from 45.9 to
50.7 on R2R-CE, and from 49.8 to 54.1 on RxR-CE. Further
reducing the mixing ratio β (rows 2-4) yields continued per-
formance improvements. Specifically, decreasing β from
0.75 to 0.25 improves SR by 8.8 points (50.7 to 59.5) on
R2R-CE and 8.5 points (54.1 to 62.6) on RxR-CE. This is
because a lower β encourages the agent to explore more di-
verse, error-prone trajectories. However, this improvement
comes at a cost. The average length of collected trajecto-
ries (#Train Step) also increases significantly, from 66 (row
2) to 128 (row 4) on R2R-CE. Not only does the training
overhead increase due to the excessively long training tra-
jectories, but so does the length of the inferred paths (#Infer
Step). These results demonstrate a clear trade-off between
success rate and trajectory length.
The “Dynamic ratio” (row 5) achieves the highest SR
7
Fixed ratio (𝛽=0.5)
Fixed ratio (𝛽=0.25)
Dynamic ratio
Fixed ratio (𝛽=0.75)
Figure 3. Visualization of the DAgger-generated trajectories. The green lines denote ground truth trajectories, and the blue lines denote
predicted trajectories.
NE↓
OS↑
SR↑
SPL↑
R2R-CE,RxR-CE
+ DAgger
+ ScaleVLN
80
73.7
2D tokens only
6.80
50.7
42.3
38.4
+ StreamVGGT [53]
6.41
54.5
45.9
41.9
+ Stream3R [22]
6.39
55.5
47.1
42.6
69.0
70
64.2
60.8
60
55.9
54.5
53.7
50
Table 5. Effect of the 3D geometry encoder on R2R-CE. All mod-
els are trained with first stage only.
45.9
41.9
40
SR
SPL
OS
30
achieves an SR of 45.9 and an SPL of 41.9. Incorporating
the DAgger data brings the most significant improvement,
boosting SR to 60.8 and SPL to 53.7. Finally, including the
ScaleVLN data further enhances performance to 64.2 SR
and 55.9 SPL.
Figure 4. Ablation study of data composition on R2R-CE. The
baseline (R2R, RxR) is progressively augmented with DAgger
data (+ DAgger) and the ScaleVLN dataset (+ ScaleVLN).
and SPL on both R2R-CE and RxR-CE, surpassing even
the best-performing constant ratio (row 4).
Crucially, it
achieves this with a much lower exploration length, reduc-
ing the #Train Step from 128 to 82 on R2R-CE, and from
160 to 121 on RxR-CE.
Effect of 3D Geometry. To evaluate the impact of 3D ge-
ometry, we compare StreamVGGT [53] and Stream3R [22].
As shown in Table 5, StreamVGGT and Stream3R yield im-
provements of 3.9% and 4.8% in SR, respectively. Although
Stream3R performs slightly better, we adopt StreamVGGT
for its lower memory cost during training.
Visualization for DAgger-generated trajectories. Figure
3 visualizes the BEV map of trajectories generated by DAg-
ger. A high fixed ratio (e.g., β = 0.75) leads to trajectories
composed predominantly of oracle actions, yielding inef-
fective error-recovery data. Conversely, a low fixed ratio
(e.g., β = 0.25) allows the agent to deviate from the expert
path frequently, but leads to long exploration paths. Our dy-
namic ratio resolves this trade-off, thus collecting valuable
error-recovery data while avoiding excessive exploration.
5. Conclusion
In this paper, we address the substantial training cost
of MLLM-based Vision-Language Navigation (VLN) by
proposing Efficient-VLN, a training-efficient VLN model.
Efficient-VLN introduces two key innovations: (1) efficient
memory representations (progressive and recursive) to mit-
igate the quadratic computational burden of extensive ob-
servations, and (2) a dynamic mixed policy to balance the
exploration-efficiency trade-off. Experiments demonstrate
that Efficient-VLN achieves new state-of-the-art perfor-
mance on both R2R-CE (62.3% SR) and RxR-CE (64.5%
SR), using merely 282 H800 GPU hours, providing a strong
and efficient baseline for this area.
Ablation Study of Data Composition. We then investi-
gate the impact of data composition. Specifically, we com-
pare the model after stage 1 (trained on R2R-CE, RxR-CE)
with models fine-tuned in stage 2 by gradually incorporat-
ing the DAgger and ScaleVLN data into the training data.
As shown in Figure 4, the baseline (“R2R-CE, RxR-CE”)
8
![image](images/pdf_image_8_1.jpeg)

![image](images/pdf_image_8_2.jpeg)

![image](images/pdf_image_8_3.jpeg)

![image](images/pdf_image_8_4.jpeg)

References
ing the 3d world into large language models. In NeurIPS,
2023. 12
[1] Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan
Huang, Keji He, and Liang Wang. Etpnav: Evolving topo-
logical planning for vision-language navigation in continu-
ous environments. TPAMI, 2023. 6
[15] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang,
Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao,
Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and
large language models with object identifiers. In NeurIPS,
2024. 12
[2] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark
Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and
Anton Van Den Hengel.
Vision-and-language navigation:
Interpreting visually-grounded navigation instructions in real
environments. In CVPR, 2018. 1, 2, 11
[16] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun
Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu,
Baoxiong Jia, and Siyuan Huang. An embodied generalist
agent in 3d world. In ICML, 2024. 12
[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki
Kawanabe. Scanqa: 3d question answering for spatial scene
understanding. In CVPR, 2022. 5, 11, 12
[17] Gabriel Ilharco, Vihan Jain, Alexander Ku, Eugene Ie, and
Jason Baldridge. General evaluation for instruction condi-
tioned navigation using dynamic time warping. 2019. 6
[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report. arXiv preprint
arXiv:2502.13923, 2025. 3, 5, 11
[18] Jacob Krantz and Stefan Lee. Sim-2-sim transfer for vision-
and-language navigation in continuous environments.
In
ECCV, 2022. 6
[19] Jacob Krantz, Erik Wijmans, Arjun Majundar, Dhruv Batra,
and Stefan Lee. Beyond the nav-graph: Vision and language
navigation in continuous environments. In ECCV, 2020. 1,
2, 4, 5, 6
[5] Kevin Chen, Junshen K Chen, Jo Chuang, Marynel V´azquez,
and Silvio Savarese. Topological planning with transformers
for vision-and-language navigation. In CVPR, 2021. 1, 2, 6
[20] Jacob Krantz, Aaron Gokaslan, Dhruv Batra, Stefan Lee, and
Oleksandr Maksymets.
Waypoint models for instruction-
guided navigation in continuous environments.
In ICCV,
2021. 1, 2, 6
[6] Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng,
Thomas H Li, Mingkui Tan, and Chuang Gan.
Weakly-
supervised multi-granularity map learning for vision-and-
language navigation.
arXiv preprint arXiv:2210.07506,
2022. 6
[21] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and
Jason Baldridge. Room-across-room: Multilingual vision-
and-language navigation with dense spatiotemporal ground-
ing. In EMNLP, 2020. 1, 2, 4, 6, 11
[7] Shizhe Chen,
Pierre-Louis Guhur,
Makarand Tapaswi,
Cordelia Schmid, and Ivan Laptev. Think global, act local:
Dual-scale graph transformer for vision-and-language navi-
gation. In CVPR, 2022. 1, 2
[22] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou,
Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai,
Chen Change Loy, and Xingang Pan. Stream3r: Scalable
sequential 3d reconstruction with causal transformer, 2025.
3, 8
[8] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu,
Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da:
Visual interactive instruction tuning for omni-3d understand-
ing reasoning and planning. In CVPR, 2024. 12
[23] Yuxing Long, Wenzhe Cai, Hongcheng Wang, Guanqi Zhan,
and Hao Dong. Instructnav: Zero-shot system for generic
instruction navigation in unexplored environment.
arXiv
preprint arXiv:2406.04882, 2024. 1, 2, 6
[9] An-Chieh Cheng, Yang Fu, Yukang Chen, Zhijian Liu, Xiao-
long Li, Subhashree Radhakrishnan, Song Han, Yao Lu, Jan
Kautz, Pavlo Molchanov, et al. 3d aware region prompted
vision language model. arXiv preprint arXiv:2509.13317,
2025. 2
[24] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao
Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated
question answering in 3d scenes. In ICLR, 2023. 5, 11, 12
[10] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian
Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin,
Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-
language-action model for navigation.
Robotics: Science
and Systems, 2025. 1, 2, 3, 4, 5, 6, 12
[25] Yuankai
Qi,
Qi
Wu,
Peter
Anderson,
Xin
Wang,
William Yang Wang, Chunhua Shen, and Anton van den
Hengel. Reverie: Remote embodied visual referring expres-
sion in real indoor environments. In CVPR, 2020. 1, 2
[11] Georgios Georgakis, Karl Schmeckpeper, Karan Wanchoo,
Soham Dan, Eleni Miltsakaki, Dan Roth, and Kostas Dani-
ilidis. Cross-modal map learning for vision and language
navigation. In CVPR, 2022. 6
[26] Sonia Raychaudhuri, Saim Wani, Shivansh Patel, Unnat Jain,
and Angel X Chang. Language-aligned waypoint (law) su-
pervision for vision-and-language navigation in continuous
environments. arXiv preprint arXiv:2109.15207, 2021. 6
[12] Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-
Opazo, and Stephen Gould. Vln bert: A recurrent vision-
and-language bert for navigation. In CVPR, 2021. 2
[27] St´ephane Ross, Geoffrey Gordon, and Drew Bagnell. A re-
duction of imitation learning and structured prediction to no-
regret online learning. In AISTATS, 2011. 5, 11
[13] Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould. Bridg-
ing the gap between learning in discrete and continuous en-
vironments for vision-and-language navigation. In CVPR,
2022. 2, 6
[28] Hao Tan, Licheng Yu, and Mohit Bansal. Learning to nav-
igate unseen environments: Back translation with environ-
mental dropout. In NAACL, 2019. 2, 5
[29] Jesse Thomason, Michael Murray, Maya Cakmak, and Luke
Zettlemoyer.
Vision-and-dialog navigation.
In CoRL.
PMLR, 2020. 1, 2
[14] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng,
Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Inject-
9
tasks. Robotics: Science and Systems, 2025. 1, 2, 3, 4, 5,
6, 11
[30] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In CVPR, 2025. 3
[44] Kaichen
Zhang,
Bo
Li,
Peiyuan
Zhang,
Fanyi
Pu,
Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan
Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Re-
ality check on the evaluation of large multimodal models. In
Findings of the Association for Computational Linguistics:
NAACL 2025, 2025. 11
[31] Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu, Mo-
hit Bansal, Stephen Gould, Hao Tan, and Yu Qiao. Scaling
data generation in vision-and-language navigation. In ICCV,
2023. 2, 5, 6, 11
[32] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and
Shuqiang Jiang. Gridmm: Grid memory map for vision-and-
language navigation. In ICCV, 2023. 2, 6
[45] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Zi-
wei Liu, and Chunyuan Li. Video instruction tuning with
synthetic data. arXiv preprint arXiv:2410.02713, 2024. 5,
11
[33] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and
Shuqiang Jiang.
Sim-to-real transfer via 3d feature
fields for vision-and-language navigation.
arXiv preprint
arXiv:2406.09798, 2024. 6
[46] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei
Wang. Towards learning a generalist model for embodied
navigation. In CVPR, 2024. 1, 2, 12
[34] Zihan Wang, Seungjun Lee, and Gim Hee Lee. Dynam3d:
Dynamic layered 3d tokens empower vlm for vision-and-
language navigation. In NeurIPS, 2025. 2, 6
[47] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang.
Learning from videos for 3d world: Enhancing mllms with
3d vision geometry priors. In NeurIPS, 2025. 2, 3
[35] Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang
Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing
Wang, Yilun Chen, et al. Streamvln: Streaming vision-and-
language navigation via slowfast context modeling. arXiv
preprint arXiv:2507.05240, 2025. 1, 2, 3, 4, 6, 11, 12
[48] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm:
Learning position-aware video representation for 3d scene
understanding. In CVPR, 2025. 2, 12
[49] Gengze Zhou, Yicong Hong, and Qi Wu. Navgpt: Explicit
reasoning in vision-and-language navigation with large lan-
guage models. In AAAI, 2024. 1, 2
[36] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan.
Spatial-mllm: Boosting mllm capabilities in visual-based
spatial intelligence. In NeurIPS, 2025. 3
[50] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang,
and Xihui Liu. Llava-3d: A simple yet effective pathway to
empowering lmms with 3d-awareness. In ICCV, 2025. 12
[37] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han,
Li Fei-Fei, and Saining Xie. Thinking in space: How mul-
timodal large language models see, remember, and recall
spaces. In CVPR, 2025. 2
[51] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan
Huang, and Qing Li. 3d-vista: Pre-trained transformer for
3d vision and text alignment. In CVPR, 2023. 12
[38] Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng,
Hongwei Fan, Jiyao Zhang, and Hao Dong.
Correctnav:
Self-correction flywheel empowers vision-language-action
navigation model. arXiv preprint arXiv:2508.10416, 2025.
2
[52] Ziyu Zhu, Xilin Wang, Yixuan Li, Zhuofan Zhang, Xiaojian
Ma, Yixin Chen, Baoxiong Jia, Wei Liang, Qian Yu, Zhi-
dong Deng, et al. Move to understand a 3d scene: Bridging
visual grounding and exploration for efficient and versatile
embodied navigation. In ICCV, 2025. 2
[39] Shuang Zeng, Dekang Qi, Xinyuan Chang, Feng Xiong,
Shichao Xie, Xiaolong Wu, Shiyi Liang, Mu Xu, and Xing
Wei.
Janusvln: Decoupling semantics and spatiality with
dual implicit memory for vision-language navigation, 2025.
3
[53] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou,
and Jiwen Lu. Streaming 4d visual geometry transformer.
arXiv preprint arXiv:2507.11539, 2025. 2, 3, 5, 8
[40] Tao Zewei and Huang Yunpeng.
Magiattention:
A
distributed attention towards linear scalability for ultra-
long context, heterogeneous mask training.
https:
//github.com/SandAI- org/MagiAttention/,
2025. 5, 11
[41] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou,
Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and
He Wang. Navid: Video-based vlm plans the next step for
vision-and-language navigation. Robotics: Science and Sys-
tems, 2024. 1, 2, 3, 4, 5, 6, 7, 11
[42] Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang
Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu,
Xingxing Li, et al. Embodied navigation foundation model.
arXiv preprint arXiv:2509.12129, 2025. 1, 6
[43] Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li,
Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng
Zhang, and He Wang.
Uni-navid: A video-based vision-
language-action model for unifying embodied navigation
10
Efficient-VLN: A Training-Efficient Vision-Language Navigation Model
Supplementary Material
1. Additional Implementation Details
#Count
#State-Action Pairs
R2R
10,819
153,693
RxR
19,894
467,807
DAgger (R2R)
10,819
218,129
DAgger (RxR)
19,672
588,589
ScaleVLN-150K
155,098
1,821,560
SQA3D
26,623
–
ScanQA
26,515
–
LLaVA-Video-178K (subset)
48,468
–
Training Details.
Our implementation is built upon the
official Qwen2.5-VL [4] codebase. To adapt this frame-
work for the VLN task, we implement a custom dataset
module that processes navigation steps sequentially, rather
than treating them as independent instances. For training
efficiency, we group navigation trajectories by length, with
trajectories in each group padded to the group’s maximum
length. Each group is then distributed across different pro-
cesses to enable multi-GPU acceleration. To enhance GPU
parallelism, we split each trajectory’s state-action pairs into
consecutive chunks, each containing 16 steps. Within each
segment, textual prompts are concatenated into a single,
flattened prompt, while block-sparse attention [40] is em-
ployed to restrict computations to their corresponding steps.
For video question-answering and video instruction tun-
ing data, we uniformly sample 16 frames from each video.
Table 6. Statistics of the combined dataset. #Count denotes the
number of samples for VLN, Video QA, and multimodal instruc-
tion tuning tasks. #State-Action Pairs refers to the number of state-
action pairs in VLN samples, where trajectories are sampled with
a stride of 4.
2. Data Statistics
Our training data statistics are presented in Table 6. The
dataset spans three categories: Vision-Language Navigation
(VLN), 3D question-answering, and video instruction tun-
ing. For VLN datasets (e.g., R2R [2], RxR [21], ScaleVLN
[31]), we convert trajectories into state-action pairs with a
stride of 4. Non-navigation tasks, including SQA3D [24],
ScanQA [3], and a subset of LLaVA-Video-178K [45], are
included to improve the model’s generalizability.
DAgger Process.
The Dataset Aggregation (DAgger)
[27] algorithm iteratively collects new trajectories and ag-
gregates them into the existing dataset to bootstrap the
model. Specifically, this process involves utilizing a mixed
policy of the learned policy and oracle policy to generate
the trajectories, which are then relabeled with expert ac-
tions. This approach has been widely adopted by recent
MLLM-based methods [35, 41, 43] to enhance error recov-
ery capabilities, typically employing only a single iteration
of data collection. Following this convention, we also adopt
a single-iteration setting.
To accommodate our model’s prediction horizon of four
actions, we adapted the standard DAgger implementation.
Specifically, at a given state, we first advance the oracle
policy by four steps to obtain the corresponding sequence
of expert actions. We subsequently revert the environment
to the original state and proceed with trajectory generation
under the mixed policy.
3. Results on 3D Question Answering
To investigate the spatial understanding capabilities of our
model, we conduct experiments on two widely used 3D
question-answering benchmarks: SQA3D [24] and ScanQA
[3]. The comparison results for SQA3D and ScanQA are
presented in Table 7 and Table 8, respectively. These results
demonstrate that our method achieves performance highly
competitive with leading methods.
Evaluation Details
For VLN, our code is built upon the
evaluation code of StreamVLN [35]. Specifically, during
the RxR evaluation, since certain samples yield ”N/A” val-
ues for SPL, we follow StreamVLN [35] to set those in-
valid metrics to 0 for metric computation. For 3D question-
answering benchmarks such as ScanQA [3] and SQA3D
[24], our evaluation code is built upon the LMMs-Eval
[44] framwork. During inference, we uniformly sample 16
frames from each scene as input.
11
Method
Venue
Test set
Avg.
What
Is
How
Can
Which
Others
Expert Models
SQA3D [24]
ICLR 2023
31.6
63.8
46.0
69.5
43.9
45.3
46.6
3D-VisTA [51]
ICCV 2023
34.8
63.3
45.4
69.8
47.2
48.1
48.5
3D Large Language Models
LEO [16]
ICML 2024
–
–
–
–
–
–
50.0
ChatScene [15]
NeurIPS 2024
45.4
67.0
52.0
69.5
49.9
55.0
54.6
LLaVA-3D [50]
ICCV 2025
–
–
–
–
–
–
55.6
Video-3D LLM [48]
CVPR 2025
51.1
72.4
55.5
69.8
51.3
56.0
58.6
2D Vision-Langage-Action Models
Efficient-VLN
–
46.9
71.8
54.4
72.5
45.6
55.1
56.2
Table 7. Performance comparison on the test set of SQA3D [24].
Method
Venue
EM
B-1
B-2
B-3
B-4
ROUGE-L
METEOR
CIDEr
Expert Models
ScanQA [3]
CVPR22
21.05
30.24
20.40
15.11
10.08
33.33
13.14
64.86
3D-VisTA [51]
ICCV23
22.4
–
–
–
10.4
35.7
13.9
69.6
3D large Language Models
3D-LLM (Flamingo) [14]
NeurIPS23
20.4
30.3
17.8
12.0
7.2
32.3
12.2
59.2
3D-LLM (BLIP2-flant5) [14]
NeurIPS23
20.5
39.3
25.2
18.4
12.0
35.7
14.5
69.4
LL3DA [8]
CVPR24
–
–
–
–
13.53
37.31
15.88
76.79
LEO [16]
ICML24
–
–
–
–
11.5
39.3
16.2
80.0
ChatScene [15]
NeurIPS24
21.62
43.20
29.06
20.57
14.31
41.56
18.00
87.70
LLaVA-3D [50]
ICCV25
27.0
–
–
–
14.5
50.1
20.7
91.7
Video-3D LLM [48]
–
30.10
47.05
31.70
22.83
16.17
49.02
19.84
102.06
2D Vision-Language-Action Models
NaviLLM [46]
CVPR24
23.0
–
–
–
12.5
38.4
15.4
75.9
NaVILA (16 frames) [10]
RSS25
27.4
–
–
–
15.2
48.3
19.6
99.8
StreamVLN (16 frames) [35]
–
28.8
–
–
–
15.7
48.3
19.8
100.2
Efficient-VLN
–
28.3
45.1
30.6
22.5
16.2
46.2
18.7
95.6
Table 8. Performance comparison on the validation set of ScanQA [3]. EM indicates exact match accuracy, and B-1, B-2, B-3, B-4 denote
BLEU-1, -2, -3, -4, respectively.
12
