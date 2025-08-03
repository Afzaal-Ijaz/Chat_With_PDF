# Chat with PDF 📄💬

A powerful Streamlit application that allows you to chat with your PDF documents using advanced AI capabilities. Upload any PDF and ask questions about its content in natural language.

# Features ✨
 
1.PDF Upload & Processing: Upload PDF files and extract text content
2.Intelligent Chat Interface: Ask questions about your PDF content in natural language
3.AI-Powered Responses: Leverages OpenAI's GPT models for accurate and contextual answers
4.Document Understanding: Uses LangChain for advanced document processing and retrieval
5.User-Friendly Interface: Clean and intuitive Streamlit web interface
6.Real-time Processing: Get instant responses to your queries

# Technology Stack 🛠️

->Frontend: Streamlit
->AI Framework: LangChain
->Language Model: OpenAI GPT
->Document Processing: PyPDF
->Vector Database: FAISS (for document embeddings)
->Python: 3.11.

# Usage 📖

.Start the application
streamlit run app.py

1. Open your browser
2. Upload a PDF
3. Click on the file uploader,select your PDF document and wait for processing to complete.
4. Start chatting,type your questions in the chat input get AI-powered answers based on your PDF content.

# Troubleshooting 🔧
Common Issues
1) API Key Error:
Ensure your OpenAI API key is correctly set
Check if you have sufficient API credits

2) PDF Processing Issues:
Ensure PDF is not password-protected
Try with a different PDF if processing fails

3) Memory Issues:
Large PDFs may require more memory
Consider splitting large documents

4) Streamlit Issues:
Clear browser cache
Restart the Streamlit server



# Happy chatting with your PDFs! 🎉

