import os
import tempfile
import pandas as pd
import gradio as gr
import re
import math
from collections import Counter
from html import escape
from pathlib import Path

# Lightweight text-similarity approximation for the 512 MiB Render Free instance.
# This is NOT the original MiniLM semantic embedding model.
_ARABIC_SYNONYMS = {
    'تصميم': 'جرافيك هوية حملة', 'جرافيك': 'تصميم هوية',
    'تصوير': 'فيديو اخراج حملة', 'اخراج': 'تصوير فيديو',
    'تمريض': 'صحي اسعاف عيادة', 'اسعافات': 'تمريض اسعاف صحي',
    'اسعاف': 'تمريض اسعافات صحي', 'صحي': 'تمريض عيادة',
    'تدريس': 'تعليم دروس تقوية', 'تعليم': 'تدريس دروس',
    'تنظيم': 'ميداني قيادة', 'ميداني': 'تنظيم قيادة',
    'انجليزية': 'ترجمة لغة', 'ترجمة': 'انجليزية لغة',
    'مواعيد': 'اوقات وقت موعد', 'موعد': 'مواعيد اوقات',
    'تواصل': 'اتصال متابعة منسق', 'التواصل': 'اتصال متابعة منسق',
    'شكر': 'تقدير تغذية راجعة اثر', 'تخصص': 'مهارات خبرة',
    'مهارات': 'تخصص خبرة', 'فرص': 'فرصة تطوع',
}

def normalize_ar(text):
    text = str(text).lower()
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = text.replace('ى', 'ي').replace('ة', 'ه')
    text = re.sub(r'[\u064b-\u065f\u0670\u0640]', '', text)
    return re.sub(r'[^\w\s]', ' ', text)

def features(text):
    tokens = normalize_ar(text).split()
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(normalize_ar(_ARABIC_SYNONYMS.get(token, '')).split())
    joined = ' '.join(expanded)
    grams = ['c:' + joined[i:i+3] for i in range(max(0, len(joined)-2))]
    return Counter(['w:' + t for t in expanded] + grams)

def similarity(a, b):
    fa, fb = features(a), features(b)
    common = sum(min(fa[k], fb[k]) for k in fa.keys() & fb.keys())
    denom = math.sqrt(sum(v*v for v in fa.values()) * sum(v*v for v in fb.values()))
    return common / denom if denom else 0.0


BASE_VOLUNTEERS = pd.DataFrame([
 ['م01','تصميم جرافيك وهوية','مساء الخميس',3],
 ['م02','محاسبة ومراجعة مالية','صباح السبت',0],
 ['م03','تمريض وتثقيف صحي','مساء أيام الأسبوع',5],
 ['م04','تصوير وإخراج فيديو','عطلة نهاية الأسبوع',1],
 ['م05','تدريس رياضيات وعلوم','مساء الأحد والثلاثاء',2],
 ['م06','قيادة وتنظيم ميداني','صباح الجمعة والسبت',7],
 ['م07','لغة إنجليزية وترجمة','أوقات مرنة',0],
 ['م08','برمجة وتطوير مواقع','مساء الخميس',1],
 ['م09','خدمة عملاء وعلاقات','صباح السبت',4],
 ['م10','تمريض وإسعافات أولية','عطلة نهاية الأسبوع',0],
], columns=['الرمز','المهارة','الأوقات المتاحة','المشاركات السابقة'])
BASE_OPPORTUNITIES = pd.DataFrame([
 ['ف01','توزيع السلال الغذائية','صباح السبت',3,'تنظيم وميداني'],
 ['ف02','تصميم وتصوير حملة التبرع','الخميس والجمعة',2,'تصميم وتصوير'],
 ['ف03','دروس تقوية للأيتام','مساء الأحد',2,'تدريس وتعليم'],
 ['ف04','عيادة الفحص الصحي','صباح السبت',2,'تمريض وإسعاف'],
 ['ف05','استقبال الوفود الأجنبية','أوقات مرنة',1,'لغة إنجليزية وترجمة'],
], columns=['رمز الفرصة','الفرصة','الموعد','العدد المطلوب','المهارة المطلوبة'])

# شروط أهلية تجريبية؛ تستلزم مراجعة بشرية ولا تُعد تحققًا من التراخيص.
ELIGIBILITY = {
 'ف01': ['تنظيم','ميداني','خدمة عملاء'],
 'ف02': ['تصميم','تصوير'],
 'ف03': ['تدريس','تعليم'],
 'ف04': ['تمريض','إسعاف'],
 'ف05': ['إنجليزية','ترجمة'],
}


def availability(available, required):
    a, r = str(available).strip(), str(required).strip()
    if not a or not r or 'مرنة' in a or 'مرنة' in r:
        return 'يحتاج تأكيد'
    if r == 'صباح السبت':
        if 'صباح' in a and 'السبت' in a:
            return 'متوافق'
        if 'عطلة نهاية الأسبوع' in a:
            return 'يحتاج تأكيد'
        return 'غير متوافق'
    if r == 'مساء الأحد':
        if 'مساء' in a and 'الأحد' in a:
            return 'متوافق'
        if 'مساء أيام الأسبوع' in a:
            return 'يحتاج تأكيد'
        return 'غير متوافق'
    if r == 'الخميس والجمعة':
