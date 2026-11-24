import os
import tempfile
import pandas as pd
import gradio as gr
from html import escape
from pathlib import Path
import numpy as np
import time
import threading

# MiniLM is downloaded at BUILD time, not during the first button click.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '2')
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'minilm'
from functools import lru_cache

_MODEL_LOCK = threading.RLock()
_MODEL_READY = threading.Event()
_MODEL_ERROR = None

@lru_cache(maxsize=1)
def _load_model_once():
    if not MODEL_DIR.is_dir() or not (MODEL_DIR / 'modules.json').is_file():
        raise RuntimeError('ملفات MiniLM المحلية غير موجودة. راجعي Build Command و download_model.py')
    print('Loading local MiniLM (no download)...', flush=True)
    started = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(2)
    model = SentenceTransformer(str(MODEL_DIR), device='cpu', local_files_only=True)
    print(f'MiniLM loaded in {time.perf_counter()-started:.1f}s', flush=True)
    return model


def get_model():
    # Prevent two concurrent requests from initializing two large model instances.
    with _MODEL_LOCK:
        try:
            model = _load_model_once()
            _MODEL_READY.set()
            return model
        except Exception as exc:
            print(f'MiniLM load failed: {exc!r}', flush=True)
            raise gr.Error(f'تعذر تجهيز MiniLM: {exc}') from exc


def warmup_model():
    try:
        print('Background MiniLM warmup started', flush=True)
        model = get_model()
        model.encode(['تطوع', 'تنظيم'], normalize_embeddings=True, show_progress_bar=False)
        # Cache baseline for scenario comparison before visitors click.
        original_metrics()
        print('MiniLM warmup and baseline ready', flush=True)
    except Exception as exc:
        print(f'Background warmup failed: {exc!r}', flush=True)


def semantic_matrix(left, right):
    """Cosine similarity of current editable text, computed on each run."""
    model = get_model()
    # One inference call for both tables; avoids repeated model overhead.
    texts = list(map(str, left)) + list(map(str, right))
    embeddings = model.encode(texts, convert_to_numpy=True,
                              normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    split = len(left)
    return np.matmul(embeddings[:split], embeddings[split:].T)


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
        if 'الخميس' in a or 'الجمعة' in a or 'عطلة نهاية الأسبوع' in a:
            return 'يحتاج تأكيد'
        return 'غير متوافق'
    # الموعد الجديد غير المدعوم لا يُعد متوافقًا تلقائيًا
    return 'يحتاج تأكيد'


def prepare(v, o):
    v = pd.DataFrame(v).copy().dropna(how='all')
    o = pd.DataFrame(o).copy().dropna(how='all')
    if len(v.columns) != 4 or len(o.columns) != 5:
        raise ValueError('تغيرت أعمدة البيانات؛ أعيدي تحميل البيانات الأصلية.')
    # Reorder by column names, not display position.
    if set(v.columns) == set(BASE_VOLUNTEERS.columns) and set(o.columns) == set(BASE_OPPORTUNITIES.columns):
        v = v.loc[:, BASE_VOLUNTEERS.columns]
        o = o.loc[:, BASE_OPPORTUNITIES.columns]
    else:
        v.columns = BASE_VOLUNTEERS.columns
        o.columns = BASE_OPPORTUNITIES.columns
    for col in ['الرمز','المهارة','الأوقات المتاحة']:
        v[col] = v[col].fillna('').astype(str).str.strip()
    for col in ['رمز الفرصة','الفرصة','الموعد','المهارة المطلوبة']:
        o[col] = o[col].fillna('').astype(str).str.strip()
    o['العدد المطلوب'] = pd.to_numeric(o['العدد المطلوب'], errors='coerce')
    if v.empty or o.empty or v[['الرمز','المهارة']].eq('').any().any() or o[['رمز الفرصة','الفرصة','المهارة المطلوبة']].eq('').any().any():
        raise ValueError('يرجى إدخال رمز ومهارة لكل متطوع، ورمز واسم ومهارة لكل فرصة.')
    if v['الرمز'].duplicated().any() or o['رمز الفرصة'].duplicated().any():
        raise ValueError('يجب ألا تتكرر رموز المتطوعين أو الفرص.')
    if o['العدد المطلوب'].isna().any() or (o['العدد المطلوب'] < 0).any() or (o['العدد المطلوب'] % 1 != 0).any():
        raise ValueError('العدد المطلوب يجب أن يكون عددًا صحيحًا غير سالب.')
    o['العدد المطلوب'] = o['العدد المطلوب'].astype(int)
    return v.reset_index(drop=True), o.reset_index(drop=True)


def calculate(v, o):
    v, o = prepare(v, o)
    # Same cosine similarity as Colab; one batched MiniLM inference for speed.
    sims = semantic_matrix(o['المهارة المطلوبة'].tolist(), v['المهارة'].tolist())
    used, rows = set(), []
    for i, opp in o.iterrows():
        candidates = []
        for j, person in v.iterrows():
            pid = person['الرمز']
            if pid in used:
                continue
            score = float(sims[i, j])
            rule = ELIGIBILITY.get(opp['رمز الفرصة'])
            if rule is not None:
                if not any(term in person['المهارة'] for term in rule):
                    continue
                eligibility_note = 'اجتاز شرط المهارة التجريبي'
            else:
                # فرص جديدة: لا نعتبر التشابه إثباتًا للأهلية
                if score < 0.65:
                    continue
                eligibility_note = 'فرصة جديدة: أهلية المهارة تحتاج مراجعة'
            av = availability(person['الأوقات المتاحة'], opp['الموعد'])
            if av == 'غير متوافق':
                continue
            candidates.append((av == 'متوافق', score, j, av, eligibility_note))
        candidates.sort(reverse=True)
        for _, score, j, av, note in candidates[:int(opp['العدد المطلوب'])]:
            person = v.iloc[j]
            used.add(person['الرمز'])
            rows.append({
                'الفرصة': opp['الفرصة'], 'المتطوع': person['الرمز'],
                'المهارة': person['المهارة'], 'تشابه AI': round(score, 3),
                'حالة الموعد': av,
                'حالة القرار': 'مرشح مبدئي' if av == 'متوافق' else 'بانتظار تأكيد الموعد',
                'ملاحظة الأهلية': note,
            })
        shortage = int(opp['العدد المطلوب']) - min(len(candidates), int(opp['العدد المطلوب']))
        if shortage:
            rows.append({'الفرصة': opp['الفرصة'], 'المتطوع': 'غير متوفر', 'المهارة': '—',
                'تشابه AI': None, 'حالة الموعد': '—', 'حالة القرار': f'نقص {shortage} متطوع',
                'ملاحظة الأهلية': 'يحتاج إلى استقطاب أو تعديل شروط الفرصة'})
    result = pd.DataFrame(rows, columns=['الفرصة','المتطوع','المهارة','تشابه AI','حالة الموعد','حالة القرار','ملاحظة الأهلية'])
    valid = result[result['المتطوع'] != 'غير متوفر']
    total = int(o['العدد المطلوب'].sum())
    metrics = {
        'المقاعد المطلوبة': total,
        'الترشيحات المبدئية': len(valid),
        'نسبة الترشيح المبدئي': round(100 * len(valid)/total, 1) if total else 0,
        'مواعيد متوافقة': int((valid['حالة الموعد'] == 'متوافق').sum()),
        'نسبة توافق المواعيد': round(100 * (valid['حالة الموعد'] == 'متوافق').sum()/total, 1) if total else 0,
        'مقاعد دون مرشحين': total - len(valid),
    }
    return result, metrics

# Do not run inference at import time: Render must open its port first.
RESULT_COLUMNS = ['الفرصة','المتطوع','المهارة','تشابه AI','حالة الموعد','حالة القرار','ملاحظة الأهلية']
BASE_RESULTS = pd.DataFrame(columns=RESULT_COLUMNS)
BASE_METRICS = {'المقاعد المطلوبة': int(BASE_OPPORTUNITIES['العدد المطلوب'].sum()),
                'الترشيحات المبدئية': 0, 'نسبة الترشيح المبدئي': 0,
                'مواعيد متوافقة': 0, 'نسبة توافق المواعيد': 0,
                'مقاعد دون مرشحين': int(BASE_OPPORTUNITIES['العدد المطلوب'].sum())}
@lru_cache(maxsize=1)
def original_metrics():
    return calculate(BASE_VOLUNTEERS, BASE_OPPORTUNITIES)[1]

# ملاحظات تدريبية افتراضية، لا تمثل أشخاصًا حقيقيين.
BASE_FEEDBACK = pd.DataFrame([
 ['ر01','الفرص غالبًا في أوقات عملي ولا أستطيع الحضور'],
 ['ر02','سجلت في الجمعية ولم يتواصل معي أحد بعد التسجيل'],
 ['ر03','أحتاج فرصًا أقرب إلى خبرتي في التصميم'],
 ['ر04','لم أعرف هل كانت مشاركتي مفيدة أو ماذا حققنا'],
 ['ر05','أرغب في التطوع لكن مواعيد الأنشطة غير واضحة'],
 ['ر06','لا يصلني شكر أو تغذية راجعة بعد النشاط'],
 ['ر07','تصلني فرص لا تناسب تخصصي في التمريض'],
 ['ر08','التواصل مع منسق الفرصة بطيء جدًا'],
], columns=['رمز الملاحظة','ملاحظة المتطوع'])

FEEDBACK_CATEGORIES = {
 'عدم ملاءمة المواعيد': 'مواعيد الأنشطة والفرص غير واضحة أو تتعارض مع أوقات عملي ولا أستطيع الحضور',
 'ضعف التواصل والمتابعة': 'سجلت في الجمعية ولم يتواصل معي أحد، التواصل مع المنسق بطيء والمتابعة ضعيفة',
 'عدم توافق المهارات': 'أحتاج فرصا تناسب خبرتي وتخصصي ومهاراتي مثل التصميم والتمريض، الفرص لا تناسب تخصصي',
 'ضعف التقدير وإظهار الأثر': 'لا يصلني شكر أو تقدير أو تغذية راجعة بعد النشاط ولا أعرف أثر مشاركتي',
}
CATEGORY_NAMES = list(FEEDBACK_CATEGORIES)
CATEGORY_TEXTS = list(FEEDBACK_CATEGORIES.values())
@lru_cache(maxsize=1)
def category_embeddings():
    return get_model().encode(CATEGORY_TEXTS, convert_to_numpy=True,
                              normalize_embeddings=True, batch_size=8)

def analyze_feedback(data):
    df = pd.DataFrame(data).copy().dropna(how='all')
    if len(df.columns) != 2:
        raise gr.Error('جدول الملاحظات يحتاج عمودين: الرمز والملاحظة.')
    df.columns = ['رمز الملاحظة','ملاحظة المتطوع']
    df['ملاحظة المتطوع'] = df['ملاحظة المتطوع'].fillna('').astype(str).str.strip()
    df = df[df['ملاحظة المتطوع'] != '']
    if df.empty:
        raise gr.Error('أضيفي ملاحظة واحدة على الأقل.')
    note_embeddings = get_model().encode(df['ملاحظة المتطوع'].tolist(), convert_to_numpy=True, normalize_embeddings=True, batch_size=8)
    sims = np.matmul(note_embeddings, category_embeddings().T)
    rows = []
    for i, (_, item) in enumerate(df.iterrows()):
        best = max(range(len(CATEGORY_NAMES)), key=lambda j: sims[i, j])
        score = float(sims[i][best])
        rows.append({
            'ملاحظة المتطوع': item['ملاحظة المتطوع'],
            'التصنيف المقترح': CATEGORY_NAMES[best] if score >= 0.32 else 'غير واضح',
            'تشابه AI': round(score, 2),
            'المراجعة': 'مراجعة بشرية',
        })
    result = pd.DataFrame(rows)
    counts = result['التصنيف المقترح'].value_counts()
    cards = ''.join(f'<div class="metric"><div class="metric-number">{int(count)}</div><div class="metric-label">{escape(str(category))}</div></div>' for category, count in counts.items())
    html = f'<div dir="rtl"><div class="metric-grid">{cards}</div><p class="fineprint">هذه تصنيفات مقترحة من نموذج تشابه دلالي، وليست تشخيصًا لأسباب التسرب أو قياسًا لانتشارها. البيانات افتراضية، ويجب مراجعة كل تصنيف قبل اتخاذ إجراء.</p></div>'
    return result, html


def retention_impact_html(before_active, after_active, before_completed, after_completed,
                          before_satisfaction, after_satisfaction, operating_cost,
                          before_retention, after_retention, data_owner):
    values = [before_active, after_active, before_completed, after_completed,
              before_satisfaction, after_satisfaction, operating_cost,
              before_retention, after_retention]
    if any(x is None or float(x) < 0 for x in values):
        raise gr.Error('أدخلي قيمًا غير سالبة لجميع المؤشرات.')
    if any(float(x) > 100 for x in [before_satisfaction, after_satisfaction, before_retention, after_retention]):
        raise gr.Error('نسب الرضا والاستبقاء يجب أن تكون بين 0 و100.')
    def fmt(x): return f'{float(x):g}'
    def row(name, a, b, unit='', kind='نتيجة'):
        diff = float(b)-float(a)
        sign = '+' if diff > 0 else ''
        return f'<tr><td>{name}</td><td>{kind}</td><td>{fmt(a)}{unit}</td><td>{fmt(b)}{unit}</td><td>{sign}{fmt(diff)}{unit}</td></tr>'
    owner = escape(str(data_owner or 'لم يُحدد بعد'))
    return f'''<div dir="rtl" class="impact">
    <h3>كيف ننتقل من الترشيحات إلى قياس النتائج؟</h3>
    <p class="fineprint">هذه <b>أرقام تعليمية افتراضية</b> لا تأتي من المطابقة الذكية، ولا تمثل نتائج حقيقية لجمعية.</p>
    <table class="impact-table"><thead><tr><th>المؤشر</th><th>المستوى</th><th>خط الأساس</th><th>بعد التجربة (افتراضي)</th><th>التغير</th></tr></thead><tbody>
    {row('المتطوعون النشطون',before_active,after_active,kind='نتيجة تشغيلية')}
    {row('الفرص المنفذة فعليًا',before_completed,after_completed,kind='مخرجات')}
    {row('استبقاء المتطوعين بعد 3 أشهر',before_retention,after_retention,' نقطة مئوية' if False else '%',kind='نتيجة')}
    {row('رضا المستفيدين',before_satisfaction,after_satisfaction,'%',kind='نتيجة')}
    </tbody></table>
    <p class="fineprint"><b>طريقة قياس الاستبقاء:</b> عدد متطوعي المجموعة نفسها الذين شاركوا مجددًا خلال 3 أشهر ÷ عدد متطوعي المجموعة عند بداية المتابعة × 100. ثبّتي تعريف المشاركة وفترة القياس.</p>
    <p><b>المخرجات:</b> الترشيحات والفرص المنفذة. <b>النتائج:</b> المشاركة الفعلية والاستمرار والرضا. <b>الأثر المجتمعي:</b> تحسن موثق ومستمر في الخدمة للمستفيدين، ولا تثبته هذه الأرقام وحدها.</p>
    <h4>هل الحل قابل للاستمرار؟</h4>
    <p><b>تكلفة تشغيل شهرية مفترضة:</b> {float(operating_cost):,.0f} ريال. <b>مسؤول تحديث البيانات:</b> {owner}.</p>
    <p>راجعي شهريًا: جودة البيانات، وموافقة المتطوعين، والقرارات البشرية، وتكلفة التشغيل مقابل الوقت الموفر وجودة الخدمة.</p>
    <p class="fineprint">لقياس أثر موثوق: قارني فترات متماثلة، ووثّقي مصدر البيانات، وراجعي العوامل الأخرى المؤثرة. الزيادة وحدها لا تثبت أن التطبيق سبب التحسن.</p>
    </div>'''


def metrics_html(metrics, baseline):
    def tile(title, val, suffix=''):
        return f'<div class="metric"><div class="metric-number">{val}{suffix}</div><div class="metric-label">{title}</div></div>'
    cards = ''.join([
        tile('المقاعد المطلوبة', metrics['المقاعد المطلوبة']),
        tile('الترشيحات المبدئية', metrics['الترشيحات المبدئية']),
        tile('نسبة الترشيح المبدئي', metrics['نسبة الترشيح المبدئي'], '%'),
        tile('نسبة توافق المواعيد', metrics['نسبة توافق المواعيد'], '%'),
        tile('المقاعد دون مرشحين', metrics['مقاعد دون مرشحين']),
    ])
    delta = round(metrics['نسبة الترشيح المبدئي'] - baseline['نسبة الترشيح المبدئي'], 1)
    sign = '+' if delta > 0 else ''
    return f'''<div dir="rtl"><div class="metric-grid">{cards}</div>
    <div class="comparison"><b>مقارنة السيناريوهات:</b> نسبة الترشيح الأصلية {baseline['نسبة الترشيح المبدئي']}% ← الحالية {metrics['نسبة الترشيح المبدئي']}% <strong>({sign}{delta} نقطة مئوية)</strong></div>
    <p class="fineprint">هذه مؤشرات ترشيح تجريبية، وليست قبولًا مؤكدًا أو أثرًا اجتماعيًا محققًا. الأهلية المهنية والتراخيص تحتاج تحققًا بشريًا.</p></div>'''


def impact_html(metrics, hours, cost):
    # سيناريو تخطيطي لا يقيس وفورات فعلية
    baseline_hours = 15.0
    estimate = max(0.0, baseline_hours * float(hours)/100.0)
    yearly = estimate * 52
    money = yearly * float(cost)
    return f'''<div dir="rtl" class="impact">
    <h3>🌱 الاستدامة والأثر — سيناريو تقديري</h3>
    <p>نفترض أن التنسيق اليدوي يستغرق <b>15 ساعة أسبوعيًا</b>، وأن نسبة الوقت الممكن توفيره هي <b>{hours:.0f}%</b> (افتراض قابل للتعديل).</p>
    <div class="metric-grid"><div class="metric"><div class="metric-number">{estimate:.1f}</div><div class="metric-label">ساعة محتمل توفيرها أسبوعيًا</div></div>
    <div class="metric"><div class="metric-number">{yearly:.0f}</div><div class="metric-label">ساعة محتمل توفيرها سنويًا</div></div>
    <div class="metric"><div class="metric-number">{money:,.0f}</div><div class="metric-label">ريال قيمة وقت تقديرية سنويًا</div></div></div>
    <p class="fineprint">ليست وفورات مقاسة. يلزم اختبار قبل/بعد، وتوثيق ساعات العمل الفعلية، وحساب تكاليف تشغيل النموذج ومراجعته، والتحقق من رضا المتطوعين واستمرار مشاركتهم. الأثر الاجتماعي لا يُستنتج من عدد الترشيحات.</p>
    <p><b>المخرجات:</b> عدد الترشيحات. <b>النتائج:</b> عدد المشاركات المنفذة وعودة المتطوعين. <b>الأثر:</b> تحسن خدمة المستفيدين على مدى زمني مناسب.</p>
    </div>'''


def export_csv(df):
    path = os.path.join(tempfile.mkdtemp(), 'athar_matching_results.csv')
    pd.DataFrame(df).to_csv(path, index=False, encoding='utf-8-sig')
    return path


def run(v, o, savings_pct, hourly_cost):
    start = time.perf_counter()
    print(f"Matching request started; model_ready={_MODEL_READY.is_set()}", flush=True)
    try:
        result, metrics = calculate(v, o)
    except ValueError as e:
        raise gr.Error(str(e))
    elapsed = time.perf_counter() - start
    print(f'Matching completed in {elapsed:.1f}s', flush=True)
    v_now, o_now = prepare(v, o)
    changed_v = sum((str(row['المهارة']) != str(BASE_VOLUNTEERS.set_index('الرمز').loc[row['الرمز'], 'المهارة'])) for _, row in v_now.iterrows() if row['الرمز'] in BASE_VOLUNTEERS['الرمز'].values)
    changed_o = sum((str(row['المهارة المطلوبة']) != str(BASE_OPPORTUNITIES.set_index('رمز الفرصة').loc[row['رمز الفرصة'], 'المهارة المطلوبة'])) for _, row in o_now.iterrows() if row['رمز الفرصة'] in BASE_OPPORTUNITIES['رمز الفرصة'].values)
    status = (f"✅ اكتملت إعادة المطابقة خلال {elapsed:.1f} ثانية (مهارات متطوعين معدلة: {changed_v}، متطلبات فرص معدلة: {changed_o}). {len(pd.DataFrame(v))} متطوعين، "
              f"{len(pd.DataFrame(o))} فرص، {metrics['الترشيحات المبدئية']} ترشيحات مبدئية. "
              "النسبة محسوبة من الترشيحات الفعلية للبيانات الحالية، دون تثبيت أي نسبة.")
    return (result[RESULT_DISPLAY], metrics_html(metrics, original_metrics()),
            impact_html(metrics, savings_pct, hourly_cost), export_csv(result), status)

# ترتيب العرض من اليمين لليسار، دون تغيير ترتيب بيانات النموذج.
VOL_DISPLAY = ['الرمز', 'المهارة', 'الأوقات المتاحة', 'المشاركات السابقة']
OPP_DISPLAY = ['رمز الفرصة', 'الفرصة', 'المهارة المطلوبة', 'الموعد', 'العدد المطلوب']
RESULT_DISPLAY = ['الفرصة', 'المتطوع', 'المهارة', 'تشابه AI', 'حالة الموعد', 'حالة القرار', 'ملاحظة الأهلية']

CSS = r'''
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;600;700;800&display=swap');
.gradio-container {max-width: 1300px !important; font-family: 'Tajawal', Tahoma, Arial, sans-serif !important;}
.gradio-container .prose, .gradio-container label {direction:rtl; text-align:right;}
.gradio-container .tab-nav {direction:rtl;}
.hero {direction:rtl; display:flex; justify-content:space-between; align-items:center; gap:16px; background:white; border:1px solid #e0ebe8; border-radius:16px; padding:16px;}
.hero img {max-width:160px; max-height:110px; object-fit:contain;}
.hero h1 {color:#005668; font-size:28px; margin:0;}
.hero p {color:#158e85; margin:6px 0;}
.metric-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:12px; margin:12px 0;}
.metric {background:#f6fbfa; border:1px solid #dcebe6; border-top:4px solid #36b6a2; border-radius:12px; padding:15px; text-align:center;}
.metric-number {font-size:27px; color:#005668; font-weight:800; direction:ltr;}
.metric-label {color:#365b61; font-size:13px; margin-top:6px;}
.comparison {background:#fffaf0; border-right:4px solid #c7a444; padding:13px; border-radius:8px; margin-top:12px;}
.fineprint {font-size:13px; color:#52646a; line-height:1.8;}
.impact {direction:rtl; text-align:right;}
#match-button {background:#005668 !important; color:white !important; font-weight:700 !important;}
/* Gradio editable dataframe: fix bidi and spacing without breaking editing */
#vol-table, #opp-table, #result-table, #feedback-table, #feedback-result {direction:rtl !important;}
#vol-table table, #opp-table table, #result-table table {direction:rtl !important; width:100%; border-collapse:separate; border-spacing:0; table-layout:auto;}
#vol-table th, #opp-table th, #result-table th {background:#eaf5f2 !important; color:#005668 !important; font-weight:800 !important; white-space:normal !important;}
#vol-table td, #opp-table td, #result-table td,
#vol-table th, #opp-table th, #result-table th {direction:rtl !important; text-align:right !important; letter-spacing:0 !important; word-spacing:normal !important; font-family:'Tajawal',Tahoma,Arial,sans-serif !important; font-size:15px !important; line-height:1.7 !important; padding:9px 12px !important; vertical-align:middle !important; min-width:95px;}
#vol-table td input, #opp-table td input,
#vol-table td textarea, #opp-table td textarea,
#vol-table [role='gridcell'], #opp-table [role='gridcell'], #result-table [role='gridcell'] {direction:rtl !important; text-align:right !important; unicode-bidi:plaintext !important; letter-spacing:0 !important; word-spacing:normal !important; font-family:'Tajawal',Tahoma,Arial,sans-serif !important; font-size:15px !important;}
#vol-table tr:nth-child(even) td, #opp-table tr:nth-child(even) td, #result-table tr:nth-child(even) td {background:#f8fbfa !important;}
#vol-table td:first-child, #opp-table td:first-child {font-weight:700; color:#005668;}
#vol-table .wrap, #opp-table .wrap, #result-table .wrap {overflow-x:auto;}

#feedback-table table, #feedback-result table {direction:rtl !important; width:100%;}
#feedback-table td, #feedback-table th, #feedback-result td, #feedback-result th {direction:rtl !important; text-align:right !important; font-family:'Tajawal',Tahoma,Arial,sans-serif !important; letter-spacing:0 !important; padding:10px !important;}

/* Arabic Gradio grid: ensure nested inputs and cell content inherit the correct font */
#feedback-table, #feedback-result {direction:rtl !important;}
#feedback-table table, #feedback-result table {direction:rtl !important; table-layout:auto !important; border-collapse:separate; border-spacing:0;}
#feedback-table th, #feedback-result th {background:#eaf5f2 !important; color:#005668 !important; font-weight:800 !important;}
#feedback-table td, #feedback-result td, #feedback-table th, #feedback-result th,
#feedback-table [role="gridcell"], #feedback-result [role="gridcell"],
#feedback-table input, #feedback-table textarea, #feedback-result input,
#feedback-result textarea, #feedback-table button, #feedback-result button,
#feedback-table [data-testid], #feedback-result [data-testid] {
 direction:rtl !important; text-align:right !important; font-family:'Tajawal',Tahoma,Arial,sans-serif !important;
 letter-spacing:normal !important; word-spacing:normal !important; font-kerning:normal !important;
 font-size:15px !important; line-height:1.8 !important; unicode-bidi:plaintext !important;
}
#feedback-table td, #feedback-result td {white-space:normal !important; overflow-wrap:anywhere !important;}
#feedback-table tr:nth-child(even) td, #feedback-result tr:nth-child(even) td {background:#f8fbfa !important;}
#feedback-table th:first-child, #feedback-result th:first-child {min-width:110px !important;}
#feedback-result th:last-child, #feedback-result td:last-child {min-width:115px !important;}
#feedback-result th:nth-child(2), #feedback-result td:nth-child(2) {min-width:160px !important;}
#feedback-result th:nth-child(3), #feedback-result td:nth-child(3) {min-width:95px !important;}
#feedback-table th:nth-child(2), #feedback-table td:nth-child(2) {min-width:420px !important;}
@media (max-width:800px) {#feedback-table th:nth-child(2),#feedback-table td:nth-child(2){min-width:260px !important;}}

.impact-table {width:100%; border-collapse:collapse; margin:14px 0;}
.impact-table td, .impact-table th {border:1px solid #dcebe6; padding:10px; text-align:right;}
.impact-table th {background:#eaf5f2; color:#005668;}
'''

# Locate logos whether they are uploaded at root or inside logos/.
def find_logo(filename):
    root = Path(__file__).resolve().parent
    for path in (root / 'logos' / filename, root / filename):
        if path.is_file():
            return str(path)
    print(f'WARNING: Optional logo missing: {filename}', flush=True)
    return None

TAKAMOL = find_logo('takamol_2026.png')
TAIBAH = find_logo('taibah_university.png')
# المسارات تُخدم عبر Gradio باستخدام gr.File، وتحتاج السماح بمجلد الشعارات
from pathlib import Path
from html import escape

def logo_src(path):
    if not path:
        return ''
    import base64, mimetypes
    with open(path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('ascii')
    mime = mimetypes.guess_type(path)[0] or 'image/png'
    return f'data:{mime};base64,{content}'

with gr.Blocks(title='أثر | المطابقة الذكية للمتطوعين') as demo:
    gr.HTML(f'''<div class="hero">
      {f'<img alt="شعار ملتقى تكامل 2026" src="{logo_src(TAKAMOL)}" />' if TAKAMOL else ''}
      <div style="text-align:center"><h1>أثر | من التطوع الذكي إلى الأثر المستدام</h1>
      <p>الابتكار والاستدامة والذكاء الاصطناعي</p><small>ملتقى تكامل 2026 • نموذج تدريبي ببيانات افتراضية</small></div>
      {f'<img alt="شعار جامعة طيبة" src="{logo_src(TAIBAH)}" />' if TAIBAH else ''}
      </div>''')
    gr.Markdown('عدّلي بيانات المتطوعين أو الفرص، ثم اضغطي **تشغيل المطابقة الذكية**. يجري تجهيز MiniLM في الخلفية بعد فتح الموقع.')
    with gr.Tabs():
        with gr.Tab('👥 المتطوعون'):
            v_input = gr.Dataframe(value=BASE_VOLUNTEERS[VOL_DISPLAY], headers=VOL_DISPLAY, interactive=True, type='pandas', label='بيانات المتطوعين', row_count=(10,'dynamic'), elem_id='vol-table')
        with gr.Tab('📋 الفرص التطوعية'):
            o_input = gr.Dataframe(value=BASE_OPPORTUNITIES[OPP_DISPLAY], headers=OPP_DISPLAY, interactive=True, type='pandas', label='بيانات الفرص', row_count=(5,'dynamic'), elem_id='opp-table')
        with gr.Tab('🎯 نتائج المطابقة'):
            results_output = gr.Dataframe(value=BASE_RESULTS[RESULT_DISPLAY], headers=RESULT_DISPLAY, interactive=False, label='المرشحون المقترحون (تتحدث عند تشغيل المطابقة)', elem_id='result-table')
            csv_output = gr.File(label='تنزيل النتائج CSV')
        with gr.Tab('💬 صوت المتطوعين'):
            gr.Markdown('**لماذا تقل المشاركة؟** عدّلي الملاحظات الافتراضية، ثم شغّلي تحليل نصي. النتائج مقترحات تحتاج مراجعة بشرية.')
            feedback_input = gr.Dataframe(value=BASE_FEEDBACK, headers=list(BASE_FEEDBACK.columns), type='pandas', interactive=True, row_count=(8,'dynamic'), label='ملاحظات افتراضية', elem_id='feedback-table')
            feedback_button = gr.Button('🧠 تصنيف ملاحظات المتطوعين', variant='primary')
            feedback_result = gr.Dataframe(label='تحليل الملاحظات — راجعي الاقتراحات قبل اتخاذ قرار', interactive=False, elem_id='feedback-result')
            feedback_summary = gr.HTML()
            feedback_button.click(fn=analyze_feedback, inputs=[feedback_input], outputs=[feedback_result,feedback_summary])
        with gr.Tab('📊 مقارنة المؤشرات'):
            metrics_output = gr.HTML(value=metrics_html(BASE_METRICS, BASE_METRICS))
        with gr.Tab('🌱 الاستدامة والأثر'):
            gr.Markdown('**افتراضات قابلة للتعديل** لتقدير وفورات الوقت، وليست قياسات فعلية.')
            savings = gr.Slider(0, 100, value=20, step=5, label='نسبة الوقت المحتمل توفيره (%)')
            hourly = gr.Number(value=40, minimum=0, label='قيمة ساعة العمل التقديرية (ريال)')
            impact_output = gr.HTML(value=impact_html(BASE_METRICS, 20, 40))
            gr.Markdown('### قياس النتائج والاستدامة — بيانات تعليمية افتراضية')
            gr.Markdown('غيّري أرقام خط الأساس وبعد التجربة، ثم اضغطي **تحديث قياس النتائج**. هذه القيم لا تأتي من نظام المطابقة.')
            with gr.Row():
                before_active = gr.Number(value=60, minimum=0, label='المتطوعون النشطون — قبل')
                after_active = gr.Number(value=75, minimum=0, label='المتطوعون النشطون — بعد (افتراضي)')
            with gr.Row():
                before_completed = gr.Number(value=5, minimum=0, label='الفرص المنفذة — قبل')
                after_completed = gr.Number(value=7, minimum=0, label='الفرص المنفذة — بعد (افتراضي)')
            with gr.Row():
                before_satisfaction = gr.Number(value=70, minimum=0, maximum=100, label='رضا المستفيدين % — قبل')
                after_satisfaction = gr.Number(value=82, minimum=0, maximum=100, label='رضا المستفيدين % — بعد (افتراضي)')
                monthly_cost = gr.Number(value=150, minimum=0, label='تكلفة تشغيل شهرية مفترضة (ريال)')
            with gr.Row():
                before_retention = gr.Number(value=40, minimum=0, maximum=100, label='الاستبقاء بعد 3 أشهر % — قبل (افتراضي)')
                after_retention = gr.Number(value=60, minimum=0, maximum=100, label='الاستبقاء بعد 3 أشهر % — بعد (افتراضي)')
                data_owner = gr.Textbox(value='منسق التطوع', label='المسؤول عن تحديث البيانات')
            impact_measure_button = gr.Button('📈 تحديث قياس النتائج والاستدامة')
            retention_output = gr.HTML(value=retention_impact_html(60,75,5,7,70,82,150,40,60,'منسق التطوع'))
            impact_measure_button.click(fn=retention_impact_html,
                inputs=[before_active,after_active,before_completed,after_completed,before_satisfaction,after_satisfaction,monthly_cost,before_retention,after_retention,data_owner],
                outputs=[retention_output])

    run_status = gr.Markdown('أول تشغيل يستخدم البيانات الأصلية. عدّلي الجداول ثم اضغطي التشغيل مجددًا لتتغير النتائج، وأعيدي القيم الأصلية لاستعادة خط الأساس. قد يتأخر أول تشغيل حتى يكتمل تجهيز MiniLM.')
    run_button = gr.Button('🧠 تشغيل المطابقة الذكية وتحديث المؤشرات', variant='primary', elem_id='match-button')
    run_button.click(fn=run, inputs=[v_input,o_input,savings,hourly], outputs=[results_output,metrics_output,impact_output,csv_output,run_status])
    gr.Markdown('**تنبيه:** نموذج تجريبي؛ التشابه الدلالي لا يثبت التأهيل، والقرارات النهائية تتطلب موافقة المتطوع والتحقق البشري.')

if __name__ == '__main__':
    print('Starting Gradio web server...', flush=True)
    demo.launch(
        server_name='0.0.0.0',
        server_port=int(os.environ.get('PORT', '10000')),
        share=False,
        theme=gr.themes.Soft(primary_hue='teal'),
        css=CSS,
        show_error=True,
        prevent_thread_lock=True,
    )
    # Bind the HTTP port first; only then initialize the heavy model.
    threading.Thread(target=warmup_model, name='minilm-warmup', daemon=True).start()
    # Keep the web service process alive without blocking the Gradio server.
    threading.Event().wait()