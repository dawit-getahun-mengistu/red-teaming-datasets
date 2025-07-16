# Dataset Description From HuggingFace

What is JailbreakBench?
Jailbreakbench is an open-source robustness benchmark for jailbreaking large language models (LLMs). The goal of this benchmark is to comprehensively track progress toward (1) generating successful jailbreaks and (2) defending against these jailbreaks. To this end, we provide the JBB-Behaviors dataset, which comprises a list of 100 distinct misuse behaviors---both original and sourced from prior work (in particular, Trojan Detection Challenge/HarmBench and AdvBench)---which were curated with reference to OpenAI's usage policies. We also provide the official JailbreakBench leaderboard, which tracks the performance of attacks and defenses on the JBB-Behaviors dataset, and a repository of submitted jailbreak strings, which we hope will provide a stable way for researchers to compare the performance of future algorithms.

Accessing the JBB-Behaviors dataset
Some of the contents of the dataset may be offensive to some readers

Each entry in the JBB-Behaviors dataset has four components:

Behavior: A unique identifier describing a distinct misuse behavior
Goal: A query requesting an objectionable behavior
Target: An affirmative response to the goal string
Category: A broader category of misuse from OpenAI's usage policies
Source: the source from which the behavior was sourced (i.e., Original, Trojan Detection Challenge 2023 Red Teaming Track/HarmBench, or AdvBench)