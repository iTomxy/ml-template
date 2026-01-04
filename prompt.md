# Find Idea-source Papers

This prompt shall guide the AI (e.g. ChatGPT) to analyse a target idea-driven paper,
and find the most directly related papers from its reference from which the target paper borrow idea.
The output is only a list of idea-source papers, w/o any explanation.

The attached pdf in the given example is: [Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](https://openreview.net/forum?id=zpGK1bOlHt).
Download & attach it together with this prompt.

```
**PROMPT:**

You are a machine learning research analyst. I will provide you with a PDF of a machine learning research paper (the "target paper"). Your task is to:

1. **Carefully read and analyze the target paper** to understand its core technical contributions, methodology, and novel techniques
2. **Identify the key "idea source papers"** from the references - these are papers where the target paper:
   - Borrows or adapts its **main technical approach or methodology**
   - Builds upon **similar algorithmic frameworks or architectural designs**
   - Extends or modifies **specific technical mechanisms or components**
   - Draws **direct methodological inspiration** (not just general motivation or background)

3. **Output ONLY a bullet list** of these idea source papers in the following format:
   * <VENUE>, <PAPER TITLE>


**Critical guidelines for identifying idea source papers:**
- **Focus on methodological similarity**: Look for papers that use similar technical approaches, not just papers that address the same problem
- **Check the "Related Work" section carefully**: Papers described as using "similar methods," "comparable approaches," or "related techniques" are often idea sources
- **Prioritize papers mentioned when describing the method**: Papers cited in the methodology/approach section that describe techniques the target paper adapts or extends
- **Include papers with shared technical components**: If the target paper uses LoRA experts, look for other papers that also use LoRA experts; if it uses curriculum learning, look for papers using curriculum learning
- **Do NOT include**:
  - General background citations or survey papers
  - Papers only cited for problem motivation
  - Dataset papers or evaluation benchmarks
  - Papers only mentioned in introduction for context
  - Papers that address the same problem but use completely different methods

**Where to look:**
1. **Related Work section**: Especially paragraphs discussing methods similar to the proposed approach
2. **Methodology section**: Papers cited when describing specific techniques or components
3. **Introduction**: Only if papers are explicitly mentioned as technical inspiration (not just motivation)

**Expected output size**: Typically 2-8 papers (focus on quality over quantity - only papers with clear methodological connections)

**Example:**
**Input:**
(the attached pdf)
**Desired Output:**
* arxiv 2024, Higher layers  need more lora experts
* arxiv 2024, Pmoe: Progressive mixture of ex perts with asymmetric transformer for continual learning
* emnlp 2024, Alphalora: Assigning lora experts based  on layer training quality
**Unesired Output:**
* ICLR 2024, Mixture of LoRA Experts (Wu et al.)
* NeurIPS 2024, COIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models (Chen et al.)
* ICLR 2021, Zero-Cost Proxies for Lightweight NAS (Abdelfattah et al.)
* ICLR 2022, LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)
**Explanation of this example:**
The attached pdf focuses on efficient weight (lora expert) allocation in weight expansion paradigm of continual learning, the challenge/problem behind is weight efficiency and task-depend weight importance determination. The papers list in Desired Output are directly related to this same problem; while the papers listed in Undesired Output are also related, those papers are not as direct as the ones in Desired Output.

In the following, I will provide another target paper pdf, please analyse it accordingly.
```
