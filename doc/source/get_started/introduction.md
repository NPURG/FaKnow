# Introduction

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library based on PyTorch, is designed to
reproduce and develop fake news detection algorithms. It includes **17 models**(refer to **Integrated Models**), covering **3 categories**:

- content based
- social context
- knowledge aware

**framework**

**Features**

- **Unified Framework**: Provide a unified interface to cover a series of algorithm development processes, including data processes, model development, training and evaluation
- **Generic Data Structure**:  Use JSON as the file format within the framework to match the format of the data crawled down, allowing the user to customize the processing of different fields.

- **Diverse Models**: Contain several representative fake news detection algorithms published in conferences or journals in recent years, including various content-based, social context-based, and knowledge-aware models.
- **Convenient Usability**: The PyTorch-based style makes it easy to use, providing rich auxiliary functions such as result visualization, log printing, and parameter saving.

- **Great Scalability**: Users simply focus on the exposed API and inherit built-in classes to reuse most of the functionality, needing to write only minimal code to meet new requirements.
