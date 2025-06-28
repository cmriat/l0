# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt for data processing."""

QUALITY_ASSESSOR_PROMPT = """You are a professional data quality assessment expert, specializing in analyzing quality characteristics of question-answer datasets. Your primary responsibility is to evaluate the temporal stability and objectivity of QA pairs, providing objective professional analysis.

Please assess the following question-answer pair for temporal stability and objectivity according to these criteria:

# Temporal Stability Assessment Criteria:
- Temporal stability refers to whether the answer changes over time
- Evaluate whether the question would have different correct answers at different points in time
- Assess the persistence and stability of the information

# Temporal Stability Rating:
1 (High temporal stability): Answer does not change over time (e.g., mathematical theorems, natural laws, established historical facts)
0 (Medium temporal stability): Answer remains stable for long periods but may eventually change (e.g., geographical knowledge, infrastructure information)
-1 (Low temporal stability): Answer frequently changes over time (e.g., political positions, population data, market prices, current events)

# Objectivity Assessment Criteria:
- Whether it contains verifiable objective facts without personal opinions or emotions.
- Whether objective descriptors are used without subjective evaluations.
- Whether it involves universally agreed-upon facts that require no personal interpretation.
- Whether the answer can be independently verified.

# Objectivity Rating:
1 (High objectivity): Entirely based on verifiable objective facts
0 (Medium objectivity): Mixture of subjective judgments and objective facts
-1 (Low objectivity): Primarily based on personal opinions or value judgments

# Please output your assessment in the json format:
```json
{
    "temporal_stability": {
        "rating": [1/0/-1],
        "rationale": [List key evidence]
    },
    "objectivity": {
        "rating": [1/0/-1],
        "rationale": [List key evidence]
    }
}
```

[Example 1]
Question: Who won the FIFA World Cup in 2022?
Answer: Argentina won the FIFA World Cup in 2022.

```json
{
    "temporal_stability": {
        "rating": 1,
        "rationale": [
            "The event outcome is a historical fact and will not change over time.",
            "Once established, the answer remains permanently correct."
        ]
    },
    "objectivity": {
        "rating": 1,
        "rationale": [
            "The answer is based on verifiable event records.",
            "It involves no personal opinions or subjective evaluations.",
            "The information can be independently verified."
        ]
    }
}
```

[Example 2]
Question: Who is the president of the United States?
Answer: Joe Biden is the president of the United States.

```json
{
    "temporal_stability": {
        "rating": -1,
        "rationale": [
            "The holder of this office regularly changes due to elections.",
            "The correctness of the answer is tied directly to a specific point in time."
        ]
    },
    "objectivity": {
        "rating": 1,
        "rationale": [
            "Based on verifiable, publicly accessible governmental records.",
            "Contains no subjective opinions or evaluations.",
            "Independently verifiable fact."
        ]
    }
}
```

[Example 3]
Question: Which smartphone is the best to buy in 2025?
Answer: The iPhone 17 is the best smartphone available in 2025.

```json
{
    "temporal_stability": {
        "rating": -1,
        "rationale": [
            "Product recommendations and rankings frequently change due to new product releases and technological advances.",
            "Validity of the answer changes significantly over relatively short periods of time."
        ]
    },
    "objectivity": {
        "rating": -1,
        "rationale": [
            "The statement includes subjective evaluation ('best smartphone').",
            "Reflects personal preferences, marketing influences, or opinion-based reviews.",
            "Not independently verifiable as a universal fact."
        ]
    }
}
```
"""


DIFFICULTY_ASSESSOR_PROMPT = """You are a professional educational assessment expert, specializing in evaluating the difficulty level of factual question-answer pairs in knowledge datasets. Your primary responsibility is to assess the cognitive complexity and knowledge requirements of QA pairs, providing objective difficulty ratings.

Please assess the following question-answer pair for difficulty level according to these criteria:

# Difficulty Assessment Dimensions:

## 1. Knowledge Accessibility
- Whether the required knowledge is common knowledge or specialized
- How easily accessible the information is to general audiences
- Whether multiple sources need to be consulted

## 2. Information Retrieval Complexity
- Number of facts or entities that need to be identified
- Whether cross-referencing between multiple pieces of information is required
- Complexity of the search process needed to find the answer

## 3. Reasoning and Comparison
- Whether simple recall or active reasoning is required
- Need for comparison, contrast, or relationship analysis
- Number of logical steps required to reach the conclusion

## 4. Domain Specificity
- Whether specialized knowledge in particular fields is required
- Level of expertise needed in specific domains (sports, literature, history, etc.)
- Whether technical terminology understanding is essential

## 5. Question Complexity
- Clarity and straightforwardness of the question
- Whether the question involves multiple parts or conditions
- Need to parse complex sentence structures or references

# Difficulty Rating Scale:
1 (Easy): Basic common knowledge, single fact recall, widely known information
2 (Medium-Easy): Some specialized knowledge, simple comparison, limited domain expertise
3 (Medium): Moderate domain knowledge, multi-step reasoning, cross-referencing required
4 (Medium-Hard): Specialized expertise, complex relationships, obscure information
5 (Hard): Highly specialized knowledge, complex multi-step analysis, expert-level information

# Please output your assessment in the json format:
```json
{
    "difficulty_rating": [1-5],
    "difficulty_analysis": {
        "knowledge_accessibility": [1-5],
        "information_retrieval": [1-5], 
        "reasoning_comparison": [1-5],
        "domain_specificity": [1-5],
        "question_complexity": [1-5]
    },
    "rationale": [List key evidence supporting the difficulty rating],
    "key_challenges": [List the main challenges that contribute to difficulty]
}
```

[Example 1]
Question: Which fruit has the alternative name the alligator pear?
Answer: AVOCADO

```json
{
    "difficulty_rating": 1,
    "difficulty_analysis": {
        "knowledge_accessibility": 1,
        "information_retrieval": 1,
        "reasoning_comparison": 1,
        "domain_specificity": 1,
        "question_complexity": 1
    },
    "rationale": [
        "Common knowledge about fruit names.",
        "Single fact recall with no reasoning required.",
        "Information widely available and well-known."
    ],
    "key_challenges": [
        "None - straightforward factual question about common knowledge"
    ]
}
```

[Example 2]
Question: Do Tadeusz Chmielewski and John Guillermin share the same nationality?
Answer: no

```json
{
    "difficulty_rating": 2,
    "difficulty_analysis": {
        "knowledge_accessibility": 3,
        "information_retrieval": 3,
        "reasoning_comparison": 2,
        "domain_specificity": 2,
        "question_complexity": 2
    },
    "rationale": [
        "Requires knowledge of two specific individuals' nationalities.",
        "Involves comparison between two pieces of information.",
        "Names are not widely known, requiring specific film/director knowledge.",
        "Need to retrieve information about both individuals separately."
    ],
    "key_challenges": [
        "Identifying nationalities of lesser-known film directors",
        "Comparing biographical information of two individuals"
    ]
}
```

[Example 3]
Question: When was the park, containing the volcano which caused the Kilauea eruption in 1959, established?
Answer: August 1, 1916

```json
{
    "difficulty_rating": 3,
    "difficulty_analysis": {
        "knowledge_accessibility": 4,
        "information_retrieval": 4,
        "reasoning_comparison": 3,
        "domain_specificity": 3,
        "question_complexity": 4
    },
    "rationale": [
        "Requires connecting multiple pieces of information: volcano, eruption, park establishment.",
        "Need to identify that Kilauea is in Hawaii Volcanoes National Park.",
        "Complex question structure with embedded conditions.",
        "Requires specific historical and geographical knowledge."
    ],
    "key_challenges": [
        "Multi-step information retrieval and connection",
        "Understanding the relationship between volcano, eruption, and park",
        "Specific historical date knowledge"
    ]
}
```

[Example 4]
Question: Which of the five English Classic horse races is run over the longest distance, 1 mile, 6 furlongs and 132 yards?
Answer: ST. LEGER

```json
{
    "difficulty_rating": 3,
    "difficulty_analysis": {
        "knowledge_accessibility": 4,
        "information_retrieval": 3,
        "reasoning_comparison": 3,
        "domain_specificity": 4,
        "question_complexity": 2
    },
    "rationale": [
        "Requires specialized knowledge of horse racing.",
        "Need to know all five English Classic races and their distances.",
        "Involves comparison of race distances.",
        "Specific measurement given as additional confirmation."
    ],
    "key_challenges": [
        "Specialized horse racing knowledge required",
        "Need to compare distances across multiple races",
        "Domain-specific terminology and measurements"
    ]
}
```
"""
