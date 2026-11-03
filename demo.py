import streamlit as st
from rag.pipeline import RagEngine

# 1. إعدادات الصفحة والتصميم العام
st.set_page_config(
    page_title="ذاكرة مصر - توثيق التراث والحرف المصرية",
    page_icon="📜",
    layout="wide"
)

# 2. إضافة تصميم خلفية بصرية وتنسيق عصري متناسق مع هوية الحرف المصرية
st.markdown("""
    <style>
    .main {
        background-color: #faf9f6;
    }
    .stChatInput {
        border-radius: 20px;
    }
    .hero-box {
        background: linear-gradient(135deg, #1b4d3e 0%, #2c6b56 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .hero-box h1 {
        font-size: 2.2rem;
        margin-bottom: 10px;
        font-weight: 700;
    }
    .hero-box p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    </style>
    
    <div class="hero-box">
        <h1>📜 ذاكرة مصر: الأرشيف الرقمي الذكي للتراث والحرف</h1>
        <p>«ما لم نوثقه اليوم قد لا نستطيع استعادته غدًا» — حكاية من المصريين... ذاكرة للأجيال</p>
    </div>
""", unsafe_allow_html=True)

# 3. تحميل محرك الـ RAG مرة واحدة مع التخزين المؤقت
@st.cache_resource
def load_rag_engine():
    return RagEngine()

engine = load_rag_engine()

# 4. الشريط الجانبي (Sidebar) لتوضيح رسالة المبادرة
with st.sidebar:
    st.title("🏛️ مبادرة ذاكرة مصر")
    st.markdown("**توثيق وحماية التراث والهوية المصرية باستخدام الذكاء الاصطناعي وتحليل البيانات.**")
    st.divider()
    st.markdown("💡 **عن النظام:**")
    st.caption("يتيح لك الاستعلام الفوري عن الحرف اليدوية، العادات، والمأثورات الشعبية الموثقة في قاعدة البيانات مستنداً إلى مصادر حقيقية.")
    st.divider()
    st.caption("تم التطوير ضمن فعاليات مبادرة مصر في ذاكرة المستقبل | تكنولوجيا بلا حدود 2026")

# 5. إدارة المحادثات
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض سجل المحادثة السابق
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # عرض المصادر المحفوظة مسبقاً في السجل إن وجدت
        if "sources" in message and message["sources"]:
            with st.expander("📚 المصادر والمراجع الموثقة لهذا الرد"):
                for src in message["sources"]:
                    pub_info = f" ({src.get('publisher', '')})" if src.get('publisher') else ""
                    st.markdown(f"- **[{src.get('n')}]** {src.get('title', '')}{pub_info}")

# 6. قسم الأسئلة المقترحة التفاعلية (تضغط عليها فتُرسل مباشرة)
st.write("✨ **أسئلة سريعة مقترحة للتجربة الفورية:**")
col1, col2, col3 = st.columns(3)

preset_query = None
with col1:
    if st.button("🏺 ما هي مراحل صناعة المشبك الدمياطي؟", use_container_width=True):
        preset_query = "ما هي مراحل صناعة المشبك الدمياطي؟"
with col2:
    if st.button("🛠️ ما هي أدوات تشكيل المشبك؟", use_container_width=True):
        preset_query = "ما هي أدوات تشكيل المشبك؟"
with col3:
    if st.button("📜 تحدث عن تاريخ الحرف التراثية بمصر", use_container_width=True):
        preset_query = "تحدث عن تاريخ الحرف التراثية بمصر"

st.divider()

# 7. معالجة الإدخال (سواء من خانة الشات أو من الأزرار التفاعلية السريعة)
user_query = st.chat_input("اكتب سؤالك هنا عن أي حرفة أو مأثور شعبي مصري...")

if preset_query:
    user_query = preset_query

if user_query:
    # حفظ وعرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # توليد الإجابة من محرك الـ RAG مع جلب المصادر والمراجع
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والاسترجاع من أرشيف ذاكرة مصر..."):
            try:
                response_data = engine.ask(user_query)
                answer_text = response_data.get("answer", "عذراً، حدث خطأ في معالجة الطلب.")
                sources = response_data.get("sources", [])
            except Exception as e:
                answer_text = f"عذراً، حدث خطأ أثناء الاتصال بالمحرك: {str(e)}"
                sources = []
            
            # عرض نص الإجابة
            st.markdown(answer_text)
            
            # عرض المصادر والمراجع الموثقة بشكل أنيق
            if sources:
                with st.expander("📚 المصادر والمراجع الموثقة لهذا الرد"):
                    for src in sources:
                        pub_info = f" ({src.get('publisher', '')})" if src.get('publisher') else ""
                        st.markdown(f"- **[{src.get('n')}]** {src.get('title', '')}{pub_info}")
            
            # حفظ المساعد وردّه مع المصادر في سجل المحادثة
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer_text,
                "sources": sources
            })