Key Libraries/Services used :
SRCC Case Studies (Data)
recursivetextsplitter (for chunking)
streamlit
google-generativeai (gemini 2.0 flash)
langchain-community 
qdrant-client (for storing)
sentence-transformers (for embeddings)

Limitations:
1) gemini 2.0 has some restrictions for free tier so around 15-20 prompts can be made per minute and few hundreds per day
2) streamlit is used for deploying hence I cant really make the website better looking

This works better than I thought mainly because of two things, 
1) proper memory/history carryforward untill the chat is over
2) query enhancification and good prompt and good dataset

https://csotgenaichatbot-siddhantdosi.streamlit.app/
Developer: Siddhant Dosi
GitHub: @siddosi
Live Application: CSOT GenAI Chatbot

