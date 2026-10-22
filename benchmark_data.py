"""
Benchmark and Sample Datasets for LingoAI Studio.
Contains sample texts in English, Hindi, and Marathi,
as well as a labeled dataset for the Evaluation module.
"""

# Sample texts for one-click testing in UI
SAMPLE_TEXTS = {
    "English": {
        "text": (
            "The Indian Space Research Organisation (ISRO) successfully launched its next-generation "
            "meteorological satellite from the Satish Dhawan Space Centre in Sriharikota yesterday. "
            "Dr. S. Somanath remarked that this technological breakthrough will significantly improve "
            "cyclone tracking, monsoon predictions, and agricultural planning across South Asia. "
            "Citizens and global scientists have praised this remarkable milestone for Indian science."
        ),
        "source_lang": "en-IN",
        "description": "Space science and technology news in English"
    },
    "Hindi": {
        "text": (
            "भारतीय प्रौद्योगिकी संस्थान (IIT) दिल्ली के शोधकर्ताओं ने ग्रामीण भारत में स्वच्छ पेयजल "
            "उपलब्ध कराने के लिए एक क्रांतिकारी सौर-संचालित जल शोधन प्रणाली विकसित की है। "
            "केंद्रीय विज्ञान और प्रौद्योगिकी मंत्री ने इस नवाचार की अत्यधिक सराहना की और कहा कि "
            "यह तकनीक लाखों परिवारों के जीवन स्तर में उल्लेखनीय सुधार लाएगी।"
        ),
        "source_lang": "hi-IN",
        "description": "Technology and healthcare development in Hindi"
    },
    "Marathi": {
        "text": (
            "महाराष्ट्र राज्य शासनाने मुंबई आणि पुण्यातील वाढत्या वाहतूक कोंडीवर मात करण्यासाठी "
            "नवीन मेट्रो मार्ग आणि पर्यावरणपूरक इलेक्ट्रिक बस सेवेचा विस्तार करण्याचा स्तुत्य निर्णय घेतला आहे. "
            "या योजनेमुळे लाखो दैनंदिन प्रवाशांचा प्रवास सुखकर होणार असून हवेतील प्रदूषण मोठ्या प्रमाणात कमी होईल, "
            "असा विश्वास पर्यावरण तज्ज्ञांनी व्यक्त केला आहे."
        ),
        "source_lang": "mr-IN",
        "description": "Urban infrastructure and environment in Marathi"
    }
}

# Labeled evaluation benchmark dataset for college exhibition demonstration
# Contains verified ground truth for Sentiment Analysis and Topic Classification
BENCHMARK_EVAL_DATA = [
    # English samples
    {
        "id": 1,
        "language": "English",
        "text": "The new renewable energy initiative by the government has received overwhelming public support and industry investment.",
        "true_sentiment": "Positive",
        "true_topic": "Business"
    },
    {
        "id": 2,
        "language": "English",
        "text": "The flight was severely delayed due to technical glitches, and the airline staff refused to provide water or refreshments.",
        "true_sentiment": "Negative",
        "true_topic": "Other"
    },
    {
        "id": 3,
        "language": "English",
        "text": "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent during its monetary policy committee meeting.",
        "true_sentiment": "Neutral",
        "true_topic": "Business"
    },
    {
        "id": 4,
        "language": "English",
        "text": "Researchers at the university published a study on the migration patterns of arctic birds over the past two decades.",
        "true_sentiment": "Neutral",
        "true_topic": "Education"
    },
    {
        "id": 5,
        "language": "English",
        "text": "Our team won the national robotics championship after months of relentless dedication and hard work!",
        "true_sentiment": "Positive",
        "true_topic": "Technology"
    },
    
    # Hindi samples
    {
        "id": 6,
        "language": "Hindi",
        "text": "इस मोबाइल फोन का कैमरा और बैटरी बैकअप बेहद शानदार है। मुझे यह खरीदकर बहुत खुशी हुई।",
        "true_sentiment": "Positive",
        "true_topic": "Technology"
    },
    {
        "id": 7,
        "language": "Hindi",
        "text": "सड़क पर भारी जलभराव और गड्ढों के कारण लोगों को घंटों जाम में फंसे रहना पड़ा, जिससे भारी आक्रोश है।",
        "true_sentiment": "Negative",
        "true_topic": "Politics"
    },
    {
        "id": 8,
        "language": "Hindi",
        "text": "संसद के शीतकालीन सत्र में कई महत्वपूर्ण विधेयकों पर विस्तार से चर्चा होने की संभावना है।",
        "true_sentiment": "Neutral",
        "true_topic": "Politics"
    },
    {
        "id": 9,
        "language": "Hindi",
        "text": "भारतीय क्रिकेट टीम ने तीसरे टेस्ट मैच में शानदार प्रदर्शन करते हुए ऐतिहासिक जीत दर्ज की।",
        "true_sentiment": "Positive",
        "true_topic": "Sports"
    },
    {
        "id": 10,
        "language": "Hindi",
        "text": "अस्पताल में आवश्यक दवाओं की कमी के चलते मरीजों और उनके परिजनों को भारी परेशानियों का सामना करना पड़ रहा है।",
        "true_sentiment": "Negative",
        "true_topic": "Healthcare"
    },

    # Marathi samples
    {
        "id": 11,
        "language": "Marathi",
        "text": "शेतकऱ्यांसाठी सुरू केलेली नवी सिंचन योजना अत्यंत फायदेशीर ठरत असून शेती उत्पादनात मोठी वाढ झाली आहे.",
        "true_sentiment": "Positive",
        "true_topic": "Business"
    },
    {
        "id": 12,
        "language": "Marathi",
        "text": "ऑनलाइन खरेदी केलेल्या उत्पादनाचा दर्जा अत्यंत निकृष्ट असून विक्रेत्याने परतावा देण्यास नकार दिला.",
        "true_sentiment": "Negative",
        "true_topic": "Business"
    },
    {
        "id": 13,
        "language": "Marathi",
        "text": "पुणे विद्यापीठाने आगामी सत्र परीक्षेचे वेळापत्रक अधिकृत संकेतस्थळावर जाहीर केले आहे.",
        "true_sentiment": "Neutral",
        "true_topic": "Education"
    },
    {
        "id": 14,
        "language": "Marathi",
        "text": "नवीन कृत्रिम बुद्धिमत्ता (AI) प्रणालीमुळे स्थानिक भाषेतील शिक्षणात मोठी क्रांती घडून येईल.",
        "true_sentiment": "Positive",
        "true_topic": "Technology"
    },
    {
        "id": 15,
        "language": "Marathi",
        "text": "राष्ट्रीय महामार्गावरील अपघातानंतर वाहतूक व्यवस्था विस्कळीत झाली असून पोलीस तपास सुरू आहे.",
        "true_sentiment": "Negative",
        "true_topic": "Other"
    }
]
