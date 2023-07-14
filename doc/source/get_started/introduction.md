# Introduction

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library based on PyTorch, is designed for
reproducing and developing fake news detection algorithms. It includes **17 models**(see at **Integrated Models**), covering **3 categories**:

- content based
- social context
- knowledge aware

**framework**

**Features**

- **Unified Framework**: provide a unified interface to cover a series of algorithm development processes, including data processing, model developing, training and evaluation
- **Generic Data Structure**:  use json as the file format read into the framework to fit the format of the data crawled down, allowing the user to feed the data into the framework with only minor processing

- **Diverse Models**: contains a number of representative fake news detection algorithms published in conferences or journals during recent years, including a variety of content-based, social context-based and knowledge aware models
- **Convenient Usability**: pytorch based style makes it easy to use with rich auxiliary functions like result visualization, log printing, parameter saving

- **Great Scalability**: users just focus on the exposed api and inherit built-in classes to reuse most of the functionality and only need to write a little code to meet new requirements
