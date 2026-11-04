# README – Benchmark Question Design and Structure

## 1. Purpose of the Benchmark Dataset

The benchmark dataset is used to evaluate and compare two question-answering approaches:

1. **Base LLM:** Answers questions directly using its existing knowledge without retrieving information from the project's knowledge base.
2. **RAG-Augmented LLM:** Retrieves relevant information from the knowledge base and uses it as context to generate an answer.

Both approaches receive the same benchmark questions. Their generated answers are evaluated against predefined ground-truth answers.

The benchmark is designed around LangChain technical documentation. Questions should test different types of information retrieval, understanding, and reasoning rather than simply asking for definitions.

The objective is to investigate whether retrieval-augmented generation improves the accuracy and reliability of answers compared with a standalone LLM.

---

## 2. Dataset Structure

The benchmark questions are stored in a JSON file.

Each question is represented as a JSON object containing three required fields and one optional field:

| Field | Description |
|---|---|
| `id` | Unique identifier assigned to each question. |
| `category` | Classification describing the type of question. |
| `question` | The actual question presented to both models. |
| `ground_truth` | The expected correct answer used during evaluation. |

### Example

```json
{
  "id": "Q001",
  "category": "Direct_Factual",
  "question": "What is the default value of the max_retries parameter in init_chat_model?",
  "ground_truth": "The default value is 6."
}
```

### Explanation

- `id`: Q001 uniquely identifies this benchmark question.
- `category`: Direct_Factual indicates that the question asks for a specific factual detail.
- `question`: Contains the exact wording submitted to the Base LLM and RAG system.
- `ground_truth`: Contains the expected answer against which the generated responses will be evaluated.

The JSON structure allows questions to be loaded programmatically using Python's `json` library.

---

## 3. Question Categories

Questions are divided into categories to evaluate different capabilities of the two approaches.

Each category targets a particular type of question-answering task.

### 3.1 Direct_Factual

**Purpose:** Tests whether the model can retrieve a specific fact or piece of information from the documentation.

These questions generally have short, clearly defined answers.

Example:

```json
{
  "id": "Q001",
  "category": "Direct_Factual",
  "question": "What is the default value of the max_retries parameter in init_chat_model?",
  "ground_truth": "The default value is 6."
}
```

Expected answer: `6`

What this tests:
- Retrieval of specific technical information.
- Identification of default parameter values.
- Accuracy of factual responses.

### 3.2 Negative_Constraint

**Purpose:** Tests whether the model can recognize information that does not exist in the documentation or whether a question contains an invalid technical concept.

Example:

```json
{
  "id": "Q002",
  "category": "Negative_Constraint",
  "question": "How do you configure the auto_quantum_optimize flag in RecursiveCharacterTextSplitter?",
  "ground_truth": "The provided context does not contain sufficient information to answer this question because the auto_quantum_optimize parameter does not exist."
}
```

The question contains a fabricated parameter.

The expected RAG behaviour is to avoid inventing configuration instructions and acknowledge that the provided documentation does not support an answer.

What this tests:
- Ability to recognize unsupported information.
- Resistance to hallucination.
- Appropriate refusal when documentation does not contain the requested information.

**Important:** Before including a negative-constraint question, verify that the parameter or concept genuinely does not exist in the relevant documentation or API version. The absence of information from the indexed knowledge base is not automatically proof that something does not exist.

### 3.3 Comparative

**Purpose:** Tests whether the model can explain differences between two related technical concepts, methods, or components.

Example:

```json
{
  "id": "Q003",
  "category": "Comparative",
  "question": "What is the primary operational difference between CharacterTextSplitter and RecursiveCharacterTextSplitter?",
  "ground_truth": "CharacterTextSplitter splits strictly on a single specified character (like newline), whereas RecursiveCharacterTextSplitter takes a list of characters (paragraphs, newlines, spaces) and tries splitting them hierarchically in order to keep semantic chunks together."
}
```

What this tests:
- Understanding of multiple technical concepts.
- Ability to identify similarities and differences.
- Ability to combine information from different parts of the documentation.

### 3.4 Code_Syntax

**Purpose:** Tests whether the model can identify programming parameters, syntax, API usage, and implementation procedures from technical documentation.

Example:

```json
{
  "id": "Q004",
  "category": "Code_Syntax",
  "question": "What parameter is used in MarkdownHeaderMetadataSplitter to define which headers to split on and their associated metadata names?",
  "ground_truth": "The headers_to_split_on parameter (accepting a list of tuples like [('#', 'Header 1'), ('##', 'Header 2')])."
}
```

What this tests:
- Identification of API parameters.
- Recognition of programming syntax.
- Understanding of code examples and their usage.

Additional questions in this category can ask about implementation steps, function calls, arguments, and code configuration.

---

## 4. Designing Benchmark Questions

Group members should follow a consistent process when creating questions.

### Step 1: Select a documentation topic

Choose a topic from the LangChain documentation included in the project's knowledge base.

Examples:
- Chat models
- Text splitters
- Document loaders
- Embeddings
- Vector stores
- Retrieval
- RAG workflows

Questions should relate to the actual documentation available in the knowledge base.

### Step 2: Identify information worth testing

Find a specific fact, code example, explanation, comparison, or procedure within the selected documentation.

Avoid writing questions based entirely on general programming knowledge.

The objective is to test how well the models answer questions using the selected technical knowledge domain.

### Step 3: Formulate the question

Write a clear question that tests one identifiable concept or task.

Good questions:
- Ask for a specific piece of information.
- Have an identifiable expected answer.
- Avoid unnecessary ambiguity.
- Use accurate technical terminology.
- Can be answered or evaluated using the selected documentation.

Avoid questions that are excessively broad or combine several unrelated topics.

### Step 4: Write the ground-truth answer

The ground truth represents the expected correct answer.

It should:
- Be factually correct.
- Be supported by the documentation.
- Include important technical details.
- Be concise but sufficiently complete.
- Avoid unnecessary information.

For comparative and procedural questions, include the important points needed for a correct answer rather than relying on a single keyword.

For negative-constraint questions, define the expected behaviour clearly.

### Step 5: Verify the answer against the documentation

Before adding the question to the dataset, check the original documentation.

Confirm that:
- The answer is supported by the source.
- The terminology and parameter names are correct.
- The question matches the documentation version used by the project.
- The expected answer does not contain unsupported assumptions.

This step is essential because an incorrect ground-truth answer would make the evaluation unreliable.

### Step 6: Add the question to the JSON file

Assign a unique question ID and the appropriate category.

Follow the existing JSON structure and formatting.

---

## 5. Guidelines for Group Members

When contributing benchmark questions, follow these rules:

1. Use the existing JSON structure.
2. Assign unique IDs in sequence (Q007, Q008, Q009, etc.).
3. Select a category that accurately describes the question.
4. Use LangChain documentation already included in the knowledge base.
5. Verify the ground-truth answer using the original documentation.
6. Do not invent API parameters, functions, or configuration options.
7. Avoid duplicate or nearly identical questions.
8. Keep questions clear and technically precise.
9. Make sure the ground truth answers the actual question.
10. Use valid JSON syntax, including commas, quotation marks, and square brackets.

Each question should ideally test a distinct aspect of the documentation.

---

## 6. Adding a New Question

Use the following template when contributing a question:

```json
{
  "id": "Q007",
  "category": "Direct_Factual",
  "question": "Write your question here.",
  "ground_truth": "Write the expected correct answer here."
}
```

For example, a question about text splitting could look like:

```json
{
  "id": "Q007",
  "category": "Direct_Factual",
  "question": "What parameter controls the maximum size of a chunk in RecursiveCharacterTextSplitter?",
  "ground_truth": "The chunk_size parameter controls the maximum size of each chunk."
}
```

The example must still be checked against the exact documentation version used by the project before being included in the final dataset.

---

## 7. Benchmark Quality and Evaluation Considerations

A well-designed benchmark should contain a variety of question types.

It should not consist entirely of questions that can be answered by identifying one parameter or copying one sentence.

The dataset should include:
- Direct factual questions.
- Questions involving unsupported or nonexistent information.
- Comparative questions.
- Questions involving code syntax and API configuration.
- Questions requiring multiple pieces of information or procedural understanding.

The ground-truth answers will later be used to evaluate the outputs generated by both systems.

Possible evaluation methods include:

| Metric | Purpose |
|---|---|
| Exact Match | Checks whether the generated answer exactly matches the expected answer. |
| Token F1 | Measures overlap between the generated answer and ground truth at the token level. |
| Semantic Similarity | Evaluates whether the generated answer conveys meaning similar to the expected answer. |
| LLM-as-a-Judge | Uses an evaluator model to assess correctness and completeness against the ground truth. |

For questions involving negative constraints, evaluation should also consider whether the model appropriately refuses to provide unsupported information rather than inventing an answer.

---

## 8. Current Dataset Categories

The initial benchmark contains the following question types:

| Question ID | Category | Main purpose |
|---|---|---|
| Q001 | Direct_Factual | Identify a default parameter value. |
| Q002 | Negative_Constraint | Recognize a nonexistent parameter and avoid hallucination. |
| Q003 | Comparative | Explain the operational difference between two text splitters. |
| Q004 | Code_Syntax | Identify an API parameter and its expected structure. |
| Q005 | Code_Syntax | Understand how models can be utilized. |
| Q006 | Code_Syntax | Understand how to build a searchable knowledge base and a minimal RAG workflow. |

Q005 and Q006 still require ground-truth answers before they are ready for evaluation.

The categories can be expanded as the benchmark grows, provided each new category has a clear purpose.

---

## 9. Final Checklist Before Submitting Questions

- [ ] Unique question ID assigned.
- [ ] Appropriate category selected.
- [ ] Question is clear and unambiguous.
- [ ] Question is related to the selected documentation.
- [ ] Ground-truth answer is verified against the source.
- [ ] No duplicate questions.
- [ ] JSON syntax is valid.
- [ ] All required fields are present.

**The main principle:** Every benchmark question should have a clear purpose, a verifiable expected answer, and a category that explains what capability it is designed to test.