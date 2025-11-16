# app.py — NO SIDEBAR VERSION (YOLO + LangChain Chatbot)
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from langchain_openai import ChatOpenAI


# =====================================================================================
# 🎯 CONFIG
# =====================================================================================
MODEL_PATH = "models/best.pt"   # model fixed
CONF_THRES = 0.25               # default confidence
st.set_page_config(layout="wide", page_title="Construction Safety Chatbot")


# =====================================================================================
# 📌 HELPERS
# =====================================================================================
def xyxy_to_array(xyxy):
    arr = np.array(xyxy).astype(float).reshape(-1)[:4]
    return arr

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0

def associate_ppe(person_boxes, helmet_boxes, vest_boxes, iou_thresh=0.15):
    persons = []
    for pb in person_boxes:
        has_helmet = any(iou(pb, hb) >= iou_thresh for hb in helmet_boxes)
        has_vest   = any(iou(pb, vb) >= iou_thresh for vb in vest_boxes)
        persons.append({
            "person_box": pb,
            "has_helmet": has_helmet,
            "has_vest": has_vest
        })
    return persons


# =====================================================================================
# 📌 LOAD YOLO MODEL
# =====================================================================================
@st.cache_resource
def load_yolo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)

model = load_yolo()


# =====================================================================================
# 📌 LOAD LLM
# =====================================================================================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)


# =====================================================================================
# 🎨 UI TITLE
# =====================================================================================
st.title("🚧 AI-Powered Safety Compliance System 🚧")
st.markdown("""Visually detects workers' PPE (Personal Protective Equipment) and delivers real-time safety advisories via an integrated intelligent chatbot
            """)
st.image("./safety.png") 


# =====================================================================================
# 📤 IMAGE UPLOADER
# =====================================================================================
uploaded_img = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(img)
    
    col1, col2 = st.columns(2)
    col1.image(img_np, caption="Original Image", use_column_width=True)

    if model:
        with st.spinner("Running PPE detection..."):
            try:
                results = model.predict(img_np, conf=CONF_THRES, imgsz=640, verbose=False)
            except:
                results = model(img_np)

            r = results[0]

            # extract class names
            try:
                names = model.model.names
            except:
                names = {}

            person_boxes = []
            helmet_boxes = []
            vest_boxes = []
            no_helmet_boxes = []
            no_vest_boxes = []

            # parse YOLO output
            for b in r.boxes:
                xy = xyxy_to_array(b.xyxy.cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                name = names.get(cls, "").lower()

                if "person" in name:
                    person_boxes.append(xy)
                elif "helmet" in name and "no" not in name:
                    helmet_boxes.append(xy)
                elif "vest" in name and "no" not in name:
                    vest_boxes.append(xy)
                elif "no-helmet" in name:
                    no_helmet_boxes.append(xy)
                elif "no-vest" in name:
                    no_vest_boxes.append(xy)

            # associate PPE per person
            persons = associate_ppe(person_boxes, helmet_boxes + no_helmet_boxes, vest_boxes + no_vest_boxes)

            # Annotate image
            ann = img_np.copy()
            for p in person_boxes:
                x1,y1,x2,y2 = map(int,p)
                cv2.rectangle(ann,(x1,y1),(x2,y2),(255,255,0),2)
            for h in helmet_boxes:
                x1,y1,x2,y2 = map(int,h)
                cv2.rectangle(ann,(x1,y1),(x2,y2),(0,255,0),2)
            for v in vest_boxes:
                x1,y1,x2,y2 = map(int,v)
                cv2.rectangle(ann,(x1,y1),(x2,y2),(0,0,255),2)

            col2.image(ann, caption="Detected PPE", use_column_width=True)

            # Summary
            total_person = len(person_boxes)
            total_helmets = len(helmet_boxes)
            total_vests = len(vest_boxes)
            missing_full_ppe = sum(1 for p in persons if not (p["has_helmet"] and p["has_vest"]))

            st.subheader("📊 Detection Summary")
            st.write({
                "persons_detected": total_person,
                "helmets_detected": total_helmets,
                "vests_detected": total_vests,
                "workers_missing_full_ppe": missing_full_ppe
            })

            st.subheader("👷 Per-Person PPE Status")
            for i,p in enumerate(persons,1):
                st.write(f"Person {i}: Helmet = {'Yes' if p['has_helmet'] else 'No'}, Vest = {'Yes' if p['has_vest'] else 'No'}")

            # ------------------ MULAI: CHAT UI ------------------
            summary_text = (
                f"Detected {total_person} persons. "
                f"Helmets: {total_helmets}, Vests: {total_vests}. "
                f"Workers missing full PPE: {missing_full_ppe}."
            )

            st.markdown("---")
            st.subheader("💬 Safety Assistant (chat)")

            # init history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # show history
            chat_box = st.container()
            with chat_box:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['text']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['text']}")

            # clear history
            if st.button("Clear history"):
                st.session_state.chat_history = []
                st.rerun()

            # chat input
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input("Ask about this detection:", key="chat_input")
                submit = st.form_submit_button("Enter")

                if submit and user_input:
                    st.session_state.chat_history.append({"role": "user", "text": user_input})

                    # build prompt
                    prompt = (
                        "You are a construction safety expert. Use the detection summary.\n\n"
                        f"{summary_text}\n\n"
                        "PER-PERSON DETAILS:\n"
                    )
                    for i, p in enumerate(persons, start=1):
                        prompt += f"Person {i}: Helmet={p['has_helmet']} Vest={p['has_vest']}\n"

                    prompt += f"\nUSER QUESTION:\n{user_input}\n\nANSWER:"

                    # get answer
                    try:
                        resp = llm.invoke(prompt)
                        answer = resp.content if hasattr(resp, "content") else str(resp)
                    except Exception as e:
                        answer = f"LLM error: {e}"

                    st.session_state.chat_history.append({"role": "assistant", "text": answer})
                    st.rerun()

            # ------------------ AKHIR CHAT UI ------------------
