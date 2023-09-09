Model Introduction
===================
We have implemented 17 false information recognition algorithm models, including content-based false information
recognition algorithms, social background based false information recognition algorithms, and knowledge perception based
false information recognition algorithms.

Content based
--------------
The content-based false information recognition algorithm utilizes textual and visual information in news content to
detect false news.

.. toctree::
   :maxdepth: 1

   model/content_based/MDFEND
   model/content_based/TEXTCNN
   model/content_based/EANN
   model/content_based/MFAN
   model/content_based/SAFE
   model/content_based/SpotFake
   model/content_based/MCAN

Social context
---------------
The false information recognition algorithm based on social background detects and distinguishes false news by
investigating social background information related to news articles, especially through dissemination on social media.

.. toctree::
   :maxdepth: 1

   model/social_context/BASE_GNN
   model/social_context/BIGCN
   model/social_context/EDDFN
   model/social_context/GCNFN
   model/social_context/GNNCL
   model/social_context/UPFD

Knowledge aware
----------------
The false information recognition algorithm based on knowledge aware is a method of detecting and distinguishing false
information by utilizing the information in the knowledge base.

.. toctree::
   :maxdepth: 1

   model/knowledge_aware/FinerFact