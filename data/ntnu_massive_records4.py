"""
Massive NTNU training records 4 (~200 Q&A pairs).

Covers varied formats (short, multi-part, list, definition, opinion,
out-of-scope, step-by-step, and conversational follow-ups) to prevent
model collapse. Topics include NTNU practical info, Taiwan daily life,
comparisons, culture shock, arts calendar, and academic tips.

Sources: NTNU official site, Taiwan government portals, general knowledge.
"""

from typing import Dict, List
from ntnu_extended_records import qa


def get_massive_records4() -> List[Dict]:
    """Generate ~200 diverse-format NTNU Q&A records.

    Returns:
        List of ChatML-format record dicts.
    """
    records: List[Dict] = []

    # =====================================================================
    # Type A: Short Q&A (~40 records) — simple question, brief answer
    # =====================================================================
    records.append(qa(
        "師大幾月開學？",
        "師大每學年第一學期約於9月中旬開學，第二學期約於2月中旬開學。"
    ))
    records.append(qa(
        "師大有游泳池嗎？",
        "有。師大和平校區體育館內設有游泳池，開放時間依體育室公告為準。"
    ))
    records.append(qa(
        "師大圖書館週末有開嗎？",
        "師大圖書館總館週六週日均開放，但週日開放時間較短。寒暑假期間服務時間另行公告。"
    ))
    records.append(qa(
        "師大附近有什麼好吃的？",
        "師大商圈（師大路、龍泉街、泰順街）有大量平價餐廳、小吃與異國料理，是臺北知名美食商圈之一。"
    ))
    records.append(qa(
        "師大有語言交換的活動嗎？",
        "師大華語文教學系與國際事務處經常舉辦語言交換活動，國語教學中心也提供語言交換媒合服務。"
    ))
    records.append(qa(
        "師大學生可以修臺大的課嗎？",
        "可以。師大、臺大、臺科大組成國立臺灣大學系統，三校學生可跨校選課、互借圖書並使用部分設施。"
    ))
    records.append(qa(
        "What is NTNU's zip code?",
        "NTNU's postal codes are: Heping Campus 106, Gongguan Campus 116, Linkou Campus 244."
    ))
    records.append(qa(
        "Does NTNU have a gym?",
        "Yes. NTNU's Heping Campus has a sports complex with a swimming pool, fitness center, basketball courts, and indoor badminton courts."
    ))
    records.append(qa(
        "師大一學期學費多少？",
        "師大為國立大學，學費依學院略有差異，約每學期新臺幣25,000–30,000元之間。實際金額以會計室公告為準。"
    ))
    records.append(qa(
        "師大有心理諮商服務嗎？",
        "有。師大學務處學生輔導中心提供免費心理諮商與資源教室服務，學生可預約個別諮商或參加團體輔導活動。"
    ))
    records.append(qa(
        "師大校園有無線網路嗎？",
        "有。師大校園覆蓋eduroam及NTNU Wi-Fi無線網路，學生以帳號密碼登入即可使用。"
    ))
    records.append(qa(
        "師大附近有捷運站嗎？",
        "和平校區鄰近捷運古亭站（橘線、綠線）；公館校區鄰近捷運公館站（綠線）；林口校區鄰近機場捷運林口站。"
    ))
    records.append(qa(
        "師大的吉祥物是什麼？",
        "師大的吉祥物是「大獅兄」，設計以臺灣獅為靈感，象徵師大精神——勇氣、活力與領導力。"
    ))
    records.append(qa(
        "Taiwan's emergency number is?",
        "Taiwan's emergency number is 119 for fire and ambulance, and 110 for police. For English-language police assistance in Taipei, call (02) 2555-6000."
    ))
    records.append(qa(
        "How do I get an EasyCard in Taipei?",
        "EasyCards can be purchased at any Taipei Metro station ticket machine or convenience stores (7-Eleven, FamilyMart). They cost NT$100 deposit plus stored value."
    ))
    records.append(qa(
        "師大路上為什麼有很多書店？",
        "師大路與周邊聚集了獨立書店、二手書店與簡體字書店，因鄰近師大與臺大，形成臺北著名的溫羅汀書店文化生活圈。"
    ))
    records.append(qa(
        "師大有農曆年假嗎？",
        "師大配合行政院人事行政總處公告，農曆除夕及春節假期放假，寒假期間包含農曆年假。"
    ))
    records.append(qa(
        "師大體育館可以對外開放嗎？",
        "部分場地（如游泳池）在非教學時段開放校外人士使用，需購票或辦理月卡。詳情請洽師大體育室。"
    ))
    records.append(qa(
        "師大學生證有什麼功能？",
        "師大學生證兼具悠遊卡功能，可用於圖書館借書、門禁進出、小額消費及搭乘大眾運輸。"
    ))
    records.append(qa(
        "What language is spoken in Taipei?",
        "Mandarin Chinese (Guoyu) is the primary language. Many young people speak basic English, and Taiwanese Hokkien is also common in daily conversation."
    ))
    records.append(qa(
        "Do I need a visa to study in Taiwan?",
        "Most international students need a Resident Visa (visa: Resident). You apply at a Taiwanese embassy or representative office in your home country before arriving."
    ))
    records.append(qa(
        "師大運動會什麼時候舉辦？",
        "師大全校運動會通常在每年3月至4月間舉辦，配合校慶系列活動。"
    ))
    records.append(qa(
        "師大有校友證嗎？如何申請？",
        "師大設有校友證（終身會員卡），畢業校友可向師大祕書室校友服務中心申請，享有圖書館借書、校內停車等優惠。"
    ))
    records.append(qa(
        "師大停車方便嗎？",
        "和平校區車位有限，建議搭乘大眾運輸。公館與林口校區停車相對方便。停車收費依總務處公告辦理。"
    ))
    records.append(qa(
        "師大校慶是幾月幾號？",
        "師大校慶日為6月5日，紀念1946年臺灣省立師範學院正式成立。百年校慶於2022年舉行。"
    ))
    records.append(qa(
        "師大有提供獎學金嗎？",
        "有。師大設有入學獎學金、書卷獎、清寒獎學金、國際學生獎學金等多種獎助學金，詳見學務處生活輔導組。"
    ))
    records.append(qa(
        "師大哪個校區有宿舍？",
        "和平校區周邊有多棟學生宿舍，林口校區亦設有宿舍。公館校區目前以校外租屋為主。"
    ))
    records.append(qa(
        "師大國語教學中心是教中文的嗎？",
        "是的。師大國語教學中心（Mandarin Training Center, MTC）成立於1956年，為全球最知名的華語教學機構之一。"
    ))
    records.append(qa(
        "What is the weather like in Taipei?",
        "Taipei has a subtropical climate: hot and humid summers (28-38°C), mild winters (12-20°C), and a rainy season from May to September."
    ))
    records.append(qa(
        "How much is a Taipei Metro ticket?",
        "Taipei Metro fares range from NT$20 to NT$65 depending on distance. Using an EasyCard gives a 20% discount compared to single-trip tickets."
    ))
    records.append(qa(
        "師大有登山社嗎？",
        "有。師大登山社是校內歷史悠久的社團之一，定期舉辦百岳攀登、古道探勘與野外安全訓練活動。"
    ))
    records.append(qa(
        "師大音樂系音樂廳可以聽音樂會嗎？",
        "可以。師大音樂系每學期舉辦大量音樂會，多數開放免費入場或索票，可關注音樂系官網節目表。"
    ))
    records.append(qa(
        "師大有暑期課程嗎？",
        "有。師大暑期開設暑期班課程，供在校生與校外人士選修，另設國際暑期學校吸引外國學生來臺學習。"
    ))
    records.append(qa(
        "Can I use my laptop in NTNU classrooms?",
        "Yes. Most NTNU classrooms have power outlets and Wi-Fi. Some lecture halls also have HDMI connections for presentations."
    ))
    records.append(qa(
        "Are there vegetarian options in NTNU cafeterias?",
        "Yes. NTNU cafeterias offer vegetarian meal options. Gongguan Campus has a dedicated vegetarian buffet, and the Heping Campus canteens label meat-free dishes."
    ))
    records.append(qa(
        "師大有籃球場嗎？",
        "有。師大和平、公館與林口校區均有室外與室內籃球場，體育館內還有標準籃球場。"
    ))
    records.append(qa(
        "師大學生可以申請交換嗎？",
        "可以。師大國際事務處與全球超過200所大學簽有交換協議，學生可申請一學期或一學年的交換計畫。"
    ))
    records.append(qa(
        "師大附近有郵局嗎？",
        "有。師大和平校區旁的師大路與和平東路交叉口即有師大郵局，方便學生辦理郵務與金融業務。"
    ))
    records.append(qa(
        "師大提供日語課程嗎？",
        "師大東亞學系及進修推廣學院開設日語課程，此外師大學生可跨校選修臺大日文系課程。"
    ))

    # =====================================================================
    # Type B: Multi-part questions (~30 records)
    # =====================================================================
    records.append(qa(
        "師大管理學院是什麼時候成立的？通過了哪些國際認證？目前有幾個系所？",
        "師大管理學院於2015年正式成立，前身為管理研究所。2023年獲得AACSB國際商管認證，為亞洲師範大學首例。目前下設企業管理學系、管理研究所（碩博士班）、國際企業管理雙碩士學位學程，以及智能與指數化投資研究中心等附屬單位。"
    ))
    records.append(qa(
        "師大藝術學院有哪些系所？棟樑廳在哪裡？師大美術館何時開幕？",
        "師大藝術學院包含美術學系、設計學系與藝術史研究所。棟樑廳為師大美術系展覽空間，位於美術系館內。師大美術館於2024年正式開幕，位於和平校區，為地下2層、地上4層的現代化美術館建築，典藏近2,000件作品。"
    ))
    records.append(qa(
        "外國學生申請師大需要準備哪些文件？語言要求是什麼？截止日期？",
        "外國學生申請師大學位學程須備妥：最高學歷證明、在校成績單（中英皆可）、推薦信2封、讀書計畫、語言能力證明（華語授課程須TOCFL A2以上或HSK四級；英語授課程須TOEFL iBT 80或IELTS 6.0）、護照影本與財力證明。秋季班申請截止約為每年3月31日，春季班約為10月31日，各系所略有不同。"
    ))
    records.append(qa(
        "師大公館校區有哪些系？如何到那裏？附近有什麼餐廳？",
        "公館校區有理學院的數學、物理、化學、資訊工程、地科等系，科技與工程學院的工業教育、電機、機電、圖文傳播等系，以及國際與社會科學學院的部分系所。交通方面可搭捷運綠線至公館站或萬隆站步行10分鐘。附近公館商圈餐飲選擇豐富，水源市場、東南亞戲院周邊皆有大量平價餐廳。"
    ))
    records.append(qa(
        "How do I register for classes at NTNU? When is the add-drop period? What happens if I fail a course?",
        "Course registration at NTNU is done through the student portal (學生資訊入口網) in three phases: early registration, online add-drop, and manual add-drop. The add-drop period is typically during the first 2 weeks of each semester. If you fail a required course, you must retake it in a later semester; elective courses can either be retaken or replaced with another elective. A failed course appears on your transcript but the GPA calculation uses the retaken grade."
    ))
    records.append(qa(
        "師大歷史最久的是哪個系？哪個系學生最多？哪個系最難考？",
        "師大歷史最悠久的系為教育學系與國文學系，自1946年省立師範學院時期即設立。學生最多的系通常是教育學系與企業管理學系。錄取分數最高的系歷年來以國文學系、英語學系、音樂學系等傳統師培重點系所為前段。"
    ))
    records.append(qa(
        "師大有幾個學院？哪些學院在公館校區？哪些在和平校區？",
        "師大目前有10個學院：教育學院、文學院、理學院、藝術學院、音樂學院、管理學院、運動與休閒學院、科技與工程學院、國際與社會科學學院、生命科學專業學院。和平校區（校本部）設有教育、文學、音樂、藝術、管理、運動與休閒學院；公館校區設有理、科技與工程、國際與社會科學學院；林口校區為僑生先修部與部分國際社科院系所。"
    ))
    records.append(qa(
        "師大圖書館總館有什麼特色？公館分館呢？音樂圖書館呢？",
        "師大總館為8層建築，教育與心理學藏書量居全國之冠，設有研究小間、多媒體中心與24小時自習室。公館分館於1991年新建，以理工與科技館藏為主。音樂圖書館位於音樂學院，為全臺唯一專門音樂圖書館，收藏樂譜、唱片與音樂文獻，並支援音樂系師生研究與演奏需求。"
    ))
    records.append(qa(
        "師大研究所考試入學方式有哪些？面試佔多少比例？何時放榜？",
        "師大研究所入學管道分為甄試入學（約每年10月報名、11月面試）與考試入學（約每年12月報名、次年2月筆試）。各系比例不同，甄試通常書面審查30%-50%、面試50%-70%；考試入學筆試40%-60%、面試40%-60%。放榜時間：甄試約12月，考試入學約3月。"
    ))
    records.append(qa(
        "師大學生可以出國交換一學期嗎？語言門檻多少？需要繳師大學費嗎？",
        "可以。師大國際事務處每年公告赴外交換申請資訊，語言門檻依交換學校地區而定：英語系國家通常須TOEFL iBT 79或IELTS 6.0；日語系國家須JLPT N3以上；華語區學校免語言成績。校級交換期間繳交師大當期學費即可，姐妹校間互惠免學費。"
    ))
    records.append(qa(
        "What is the Taiwan High Speed Rail? How fast does it go? How much is Taipei to Kaohsiung?",
        "The Taiwan High Speed Rail (THSR) connects Taipei and Kaohsiung in about 1 hour 45 minutes, with a top operating speed of 300 km/h. A standard ticket from Taipei Main Station to Zuoying (Kaohsiung) costs NT$1,490 one-way. Discounts are available for early-bird reservations (早鳥優惠), student passes, and EasyCard holders on short-distance routes."
    ))
    records.append(qa(
        "師大宿舍可以住幾個學期？費用多少？幾月抽籤？",
        "師大學生宿舍保障入住年限依宿舍類型與學生身份而異：大學部一般保障2-4學期，研究生無保障需每年申請。住宿費每學期約NT$8,000–NT$15,000。宿舍抽籤申請通常在每年5月（次學年）與12月（下學期）公告，相關資訊公布於學務處住宿組官網。"
    ))
    records.append(qa(
        "師大有哪些外國學生社團？國際學生如何參與？學伴制度如何運作？",
        "師大設有國際學生會（International Student Association）、各國學生聯誼會（馬來西亞、印尼、越南等）。國際學生可透過國際事務處公告的活動報名參加。學伴制度（Buddy Program）由國際事務處媒合，本地生與國際生配對，協助外國同學適應校園與在臺生活。"
    ))
    records.append(qa(
        "師大校內打工機會多嗎？工讀金時薪多少？國際生可以打工嗎？",
        "師大提供校內圖書館、各行政單位、系辦與研究計畫工讀機會。2026年基本工資為每小時NT$190，校內工讀時薪約NT$190–NT$250。外國學生持有效居留證與工作許可（向勞動部申請）可在校內工讀，每週上限20小時（寒暑假不受限）。"
    ))
    records.append(qa(
        "師大有哪些國際雙聯學位？申請資格是什麼？",
        "師大與海外多校合作雙聯學位（2+2學士、1+1碩士等），學系含管理、教育、音樂、華語文教學等領域。申請資格：在校成績達系所規定標準（通常GPA 3.3以上）、語言能力證明（TOEFL iBT 90或IELTS 6.5）、兩校指導教授同意書。完成後可同時獲得師大與合作校之學位。"
    ))
    records.append(qa(
        "師大學生餐廳有哪些選擇？素食者有什麼選擇？平均一餐多少錢？",
        "和平校區學生餐廳（地餐）、公館校區學生餐廳與林口校區餐廳提供多元餐點，包含中式自助餐、麵食、西式簡餐、日式料理等。素食者可選擇校內素食自助餐或專屬素食窗口，多處餐廳另有蔬食選項。平均一餐約NT$60–NT$120。師大商圈則提供更多元的餐飲選擇。"
    ))
    records.append(qa(
        "How do I open a bank account in Taiwan as an international student? What documents do I need?",
        "To open a bank account in Taiwan, bring your passport, Alien Resident Certificate (ARC), student ID card, and a tax ID number. Most banks near NTNU (e.g., Bank of Taiwan, CTBC, Cathay United Bank) accept student applications. The process takes about 30 minutes. A Taiwanese phone number is helpful but not mandatory. Some banks require a minimum deposit of NT$1,000."
    ))
    records.append(qa(
        "師大哪些系所有全英語學位學程？申請條件？學費較高嗎？",
        "師大全英語學士學位學程包括：全球研究全英語學士學位學程（GIP）、物理學全英語學士學位學程（PEP）、教育跨域全英語學士學位學程（EIP）等。申請須英文檢定（TOEFL iBT 80或IELTS 6.0），學雜費與一般中文學程相同，部分學程可能有語言課程相關費用。"
    ))
    records.append(qa(
        "師大附近有哪些圖書館可以自習？哪個開最晚？需要學生證嗎？",
        "師大校內圖書館總館設有24小時自習室，刷學生證進出。附近還有臺大圖書館（需辦證或臺大系統跨校借書證）、市立圖書館大安分館、溫州街的獨立書店自習空間等。市立圖書館免費進入，不需師大學生證。"
    ))
    records.append(qa(
        "師大美術館建築有什麼特色？目前展什麼？門票多少？",
        "師大美術館由建築師陳瑞憲設計，地下2層、地上4層，採清水模與玻璃帷幕交織的現代風格，以「與自然對話」為設計理念。展覽以師大美術系師生創作、臺灣現當代藝術及國際交流展為主。門票：全票NT$100，師生與校友免費。"
    ))
    records.append(qa(
        "師大校史館在哪裡？裡面有什麼？如何參觀？",
        "師大數位校史館位於和平校區圖書館總館6樓（實體展區），同時有線上資料庫。館藏包含百年歷史文件、老照片、校歌版本、臺北高等學校時期文物與各時期制服。實體展區於圖書館開放時間免費參觀，團體導覽可預約。"
    ))
    records.append(qa(
        "師大語言中心有什麼語言課程？費用多少？如何報名？",
        "師大國語教學中心提供華語課程（團體班、個人班、線上課），另有英語、日語、韓語、法語、德語等多國語言推廣班。華語團體班每期約NT$15,000–NT$30,000（依時數），推廣班各語言約NT$4,000–NT$10,000。線上報名可至國語教學中心或進修推廣學院官網。"
    ))
    records.append(qa(
        "師大大學部有哪些特殊選才管道？原住民、體育、資優等？",
        "師大大學部入學管道除一般學測分發（繁星推薦、個人申請、考試分發）外，另有特殊選才（拾穗計畫）招收特殊才華或逆境學生、原住民專班、運動績優學生單獨招生、以及各類資優保送甄試。各管道名額與時程依教務處公告。"
    ))
    records.append(qa(
        "師大附近有哪些醫院或診所？學生看醫生方便嗎？",
        "師大和平校區附近有國立臺灣師範大學健康中心（校內初診）、臺大醫院（捷運兩站）、國泰綜合醫院、以及和平東路段多家診所（家醫、牙醫、耳鼻喉科）。持健保卡看診掛號費約NT$150–NT$400。國際學生則建議投保學生保險。"
    ))
    records.append(qa(
        "師大有哪些學術寫作資源？論文格式要求？如何找指導教授？",
        "師大圖書館提供論文寫作諮詢與Turnitin原創性比對服務。學位論文格式以APA（教育、心理、社會科學）、MLA（文學）或Chicago（歷史）為主流，各系所略有不同。找指導教授建議：先閱讀系上教授研究方向與著作，主動Email預約面談，準備研究構想。"
    ))
    records.append(qa(
        "What are the public holidays in Taiwan? Do NTNU classes stop?",
        "Taiwan's public holidays include: Republic Day (Jan 1-2), Lunar New Year (~5 days in Jan/Feb), Peace Memorial Day (Feb 28), Children's Day (Apr 4), Tomb Sweeping Day (Apr 5), Dragon Boat Festival (lunar May 5), Mid-Autumn Festival (lunar Aug 15), National Day (Oct 10). NTNU follows government announcements and suspends classes on these days."
    ))
    records.append(qa(
        "師大申請減免學雜費的條件是什麼？需要準備什麼？何時申請？",
        "符合教育部規定之低收入戶、中低收入戶、特殊境遇家庭子女或身心障礙學生可申請學雜費減免。需準備戶籍謄本、所得證明、相關身份證明文件等，於每學期註冊前（約8月、1月）向學務處生活輔導組辦理。"
    ))
    records.append(qa(
        "師大體育課可以選什麼項目？必修幾學期？可以抵免嗎？",
        "師大大學部體育為必修4學期（大一、大二各2學期），可選修項目包含籃球、排球、羽球、桌球、游泳、太極拳、有氧舞蹈、瑜珈、高爾夫等。轉學生或特殊情況可申請抵免，須檢附前校成績單經體育室核可。"
    ))

    # =====================================================================
    # Type C: List / Enumeration (~30 records)
    # =====================================================================
    records.append(qa(
        "列出臺灣五個國家公園",
        "1. **墾丁國家公園**（屏東）—臺灣首座國家公園，以珊瑚礁與海洋生態聞名。\n2. **玉山國家公園**（南投／花蓮／嘉義）—東亞最高峰玉山所在。\n3. **太魯閣國家公園**（花蓮）—大理石峽谷與斷崖景觀。\n4. **雪霸國家公園**（苗栗／臺中）—雪山與大霸尖山。\n5. **陽明山國家公園**（臺北）—火山地形與溫泉。"
    ))
    records.append(qa(
        "師大最知名的十位校友有哪些？",
        "師大知名校友橫跨各界：1. 李遠哲（諾貝爾化學獎得主，曾任教）2. 余光中（文學大師）3. 梁實秋（文學大師）4. 羅大佑（音樂人）5. 陳綺貞（創作歌手）6. 張雨生（音樂人）7. 林懷民（雲門舞集創辦人，曾就讀）8. 白先勇（作家）9. 吳念真（導演、作家）10. 陶晶瑩（主持人、歌手）。"
    ))
    records.append(qa(
        "What are three must-try Taiwanese night markets in Taipei?",
        "1. **Shilin Night Market** — largest in Taipei, famous for oyster omelets, fried chicken fillets (超大雞排), and bubble tea.\n2. **Raohe Night Market** — known for pepper pork buns (胡椒餅), grilled squid, and herbal braised eggs.\n3. **Ningxia Night Market** — smaller but authentic, specializing in Taiwanese classics like Lu-wei (滷味), stinky tofu, and taro balls (芋丸)."
    ))
    records.append(qa(
        "師大週邊有哪些書店？",
        "師大周邊書店文化豐富：1. 水準書局（師大路，以折扣書聞名）2. 唐山書店（羅斯福路，人文社科）3. 古今書廊（二手書）4. 茉莉二手書店（師大店）5. 臺灣的店（臺灣研究專書）6. 小高的店（簡體字書）。"
    ))
    records.append(qa(
        "列出師大校園內的公共藝術作品",
        "師大校園公共藝術包括：1. 太極銅質雕像（北京大學贈）2. 自由之鐘（和平校區）3. 師大美術館戶外雕塑群4. 學術大道裝置藝術5. 紅樓前日治時期石柱紀念物6. 文薈廳前廣場地景藝術。"
    ))
    records.append(qa(
        "師大UNIQLO聯名T恤有哪些款式？",
        "師大曾與UNIQLO推出UT系列聯名T恤，款式以師大紅、師大藍為基調，設計元素包括：校徽、木鐸圖案、阿勃勒花、和平校區紅樓建築剪影及「NTNU」字樣，為百年校慶系列商品之一。"
    ))
    records.append(qa(
        "What are the top 3 things to do in Taipei for NTNU international students?",
        "1. **Explore Daan Park and Yongkang Street** — A 15-minute walk from Heping Campus, great for relaxing and trying famous xiaolongbao (soup dumplings) at Din Tai Fung or local eateries.\n2. **Visit Maokong by gondola** — Take the Maokong Gondola from Taipei Zoo for stunning city views and tea houses serving Tieguanyin oolong tea.\n3. **Ride a YouBike along the riverside bike paths** — YouBike stations are all over the city; try the Keelung River bike path from Dazhi to Guandu for sunset views."
    ))
    records.append(qa(
        "師大有哪些咖啡廳適合讀書？",
        "師大周邊適合讀書的咖啡廳：1. 路燈咖啡（師大路，插座多）2. 極簡咖啡（師大路巷弄，安靜）3. 咖啡館（Cafe' a' la mode，適合久坐）4. 小米酒咖啡館（溫州街）5. 早秋咖啡（浦城街，24小時營業）。"
    ))
    records.append(qa(
        "師大運動代表隊有哪些？",
        "師大運動代表隊包括：男籃、女籃、男排、女排、足球、羽球、桌球、網球、田徑、游泳、跆拳道、柔道、射箭、擊劍、體操等。其中男排與女籃為全國甲組常勝軍。"
    ))
    records.append(qa(
        "What documents do I need for a Taiwan ARC (Alien Resident Certificate)?",
        "Required documents for ARC application: 1. Passport (valid for at least 6 months) 2. Completed ARC application form 3. 2 color photos (2-inch) 4. Resident visa 5. Official admission letter from NTNU 6. Proof of financial means 7. Health check report 8. Proof of address in Taiwan 9. Application fee (NT$1,000 for 1-year, NT$2,000 for 3-year). Apply at the National Immigration Agency within 15 days of arrival."
    ))
    records.append(qa(
        "師大校內有哪些便利商店？",
        "師大校園內與周邊便利商店：1. 7-Eleven師大門市（和平校區門口）2. 全家便利商店（師大路、圖書館校區側門）3. OK超商（公館校區內）4. 萊爾富（林口校區內）。24小時營業，全年無休。"
    ))
    records.append(qa(
        "列出台北三個必去的博物館",
        "1. **國立故宮博物院**（士林）—世界級中華文物收藏，翠玉白菜、肉形石為鎮館之寶。\n2. **臺北當代藝術館**（中山區）—前身為建成小學校舍，現展當代藝術。\n3. **國立臺灣博物館**（228公園內）—臺灣最老博物館，自然史與人類學展覽。"
    ))
    records.append(qa(
        "師大附近有哪些YouBike站？",
        "師大周邊YouBike站：1. 師大和平校區門口 2. 師大圖書館校區側門 3. 捷運古亭站2號出口 4. 龍泉市場（師大夜市口）5. 大安森林公園站。YouBike前30分鐘騎乘費用：會員NT$5（學生卡）或NT$10（一般卡）。"
    ))
    records.append(qa(
        "What types of Taiwanese street food should every NTNU student try?",
        "Every NTNU student should try these 6 Taiwanese street foods: 1. Bubble tea (珍珠奶茶) — invented in Taiwan 2. Braised pork rice (滷肉飯) — the national comfort food 3. Beef noodle soup (牛肉麵) — hearty and warming 4. Oyster omelet (蚵仔煎) — signature night market snack 5. Stinky tofu (臭豆腐) — acquired taste, deeply loved 6. Scallion pancake (蔥抓餅) — crispy, flaky, and cheap."
    ))
    records.append(qa(
        "師大多元文化週有哪些攤位活動？",
        "師大國際文化節各國攤位活動：1. 美食攤（日本章魚燒、韓國辣炒年糕、印尼炒麵、馬來西亞沙嗲）2. 文化體驗（書法、摺紙、茶道、服飾試穿）3. 遊戲攤（各國傳統遊戲）4. 語言交換配對。每年約40-50國學生設攤。"
    ))
    records.append(qa(
        "師大畢業條件有哪些？",
        "師大學士班畢業條件包括：1. 修畢各系規定畢業學分（通常128-148學分）2. 必修科目全部通過 3. 體育必修4學期 4. 通識教育核心課程（約28-32學分，涵蓋人文、社會、自然領域）5. 英文畢業門檻（依各系規定）6. 服務學習時數 7. 各系規定之專業證照、專題、實習或畢業論文。"
    ))
    records.append(qa(
        "師大英語授課通識課程有哪些類別？",
        "師大英語授課通識課程分為四大類：1. 人文與藝術（如Global Cinema、Introduction to Western Art）2. 社會科學（如International Relations、Taiwan and the World）3. 自然與科技（如Climate Change、Food Science）4. 跨域整合（如Sustainability、Cultural Heritage）。每學期開設40門以上EMI通識課。"
    ))
    records.append(qa(
        "師大就學貸款如何申請？需要準備什麼？",
        "就學貸款申請程序與文件：1. 至學務處生活輔導組領取申請表或至臺灣銀行網站下載 2. 備妥戶籍謄本、所得證明、註冊繳費單 3. 臺灣銀行對保 4. 將銀行核章之申請書繳回學校。申請期限約為每年8月及1月註冊前。在學期間利息由教育部補貼。"
    ))
    records.append(qa(
        "臺北捷運有哪些路線？",
        "臺北捷運路線：1. 文湖線（棕線）—南港展覽館至動物園 2. 淡水信義線（紅線）—淡水至象山 3. 松山新店線（綠線）—松山至新店 4. 中和新蘆線（橘線）—迴龍／蘆洲至南勢角 5. 板南線（藍線）—南港展覽館至頂埔 6. 環狀線（黃線）—大坪林至新北產業園區。師大由古亭站（橘線、綠線）與公館站（綠線）服務。"
    ))
    records.append(qa(
        "師大有哪些特色課程或學程？",
        "師大特色課程與學程：1. 師培教育學程（中等學校教師資格）2. 全英語授課學程（GIP、PEP等）3. 跨域科技產業創新研究所（AI、綠能）4. 學習科學學士學位學程 5. 生物多樣性國際博士學位學程 6. 大學社會責任實踐（USR）學程 7. 東亞研究學分學程 8. 音樂治療學分學程。"
    ))
    records.append(qa(
        "師大學生常見的休閒活動有哪些？",
        "師大學生常見休閒活動：1. 逛師大夜市 2. 大安森林公園慢跑或野餐 3. 永康街吃美食 4. 河濱自行車道騎Ubike 5. 公館商圈逛街 6. 古亭二手書店挖寶 7. 陽明山踏青 8. 各系學會與社團活動。"
    ))
    records.append(qa(
        "What smartphone apps are essential for living in Taipei?",
        "Essential apps for Taipei: 1. **EasyWallet** — manage EasyCard balance and top up 2. **YouBike** — find bikes and stations 3. **Google Maps** — reliable MRT/bus directions 4. **Foodpanda / Uber Eats** — food delivery 5. **Line** — the default messaging app in Taiwan 6. **BusTracker** (公車動態) — real-time bus arrival 7. **Taiwan Weather** — accurate hourly forecasts."
    ))
    records.append(qa(
        "師大有哪些校園安全設施？",
        "師大校園安全設施：1. 緊急通報電話（和平02-77493123、公館02-77496666）2. 校園監視系統 3. 緊急照明與逃生指示 4. 校安中心24小時值勤 5. 夜間護送服務（校園安全專車或陪同）6. AED自動體外心臟除顫器（各館舍入口）7. 消防演練與防災教育。"
    ))
    records.append(qa(
        "師大跨校選課的流程和限制是什麼？",
        "師大學生跨校選課（臺大系統）：1. 至本校教務處網站填寫跨校選課申請表 2. 經系所主管與教務處核章 3. 持申請表至對方學校教務處辦理 4. 開學後加退選期間完成。限制：每學期跨校選課以6學分為上限，須事先確認是否計入畢業學分。"
    ))
    records.append(qa(
        "師大附近有哪些運動場地？",
        "師大周邊運動場地：1. 大安運動中心（和平東路，步行10分鐘）—游泳池、健身房、羽球場 2. 臺大綜合體育館 3. 客家文化主題公園（跨提自行車道）4. 中正紀念堂廣場（跑步、滑板）5. 青年公園（棒球場）。校內體育館、籃球場、排球場則供師生優先使用。"
    ))
    records.append(qa(
        "師大有哪些心理健康資源？",
        "師大心理健康資源：1. 學輔中心個別諮商（免費，每學期可預約6-8次）2. 團體諮商（壓力管理、人際關係、情緒調適）3. 心理測驗施測與解釋 4. 心理健康講座與工作坊 5. 24小時安心專線 6. 校外轉介醫療資源（臺北市立聯合醫院松德院區等）。"
    ))
    records.append(qa(
        "師大校園內有哪些紀念性建築或景觀？",
        "師大校園紀念性建築與景觀：1. 行政大樓（市定古蹟，1929年）2. 文薈廳（原高校禮堂）3. 普字樓（日治校舍）4. 梁實秋故居 5. 劉真故居 6. 自由之鐘 7. 太極雕像 8. 阿勃勒樹道 9. 百年校慶紀念碑 10. 前校長銅像（劉真、郭為藩等）。"
    ))

    # =====================================================================
    # Type D: Definition / Explanation (~30 records)
    # =====================================================================
    records.append(qa(
        "什麼是「大學社會責任實踐計畫」（USR）？師大有參與嗎？",
        "大學社會責任實踐計畫（University Social Responsibility, USR）是教育部推動的政策，鼓勵大學運用專業知識與資源，回應在地社區與社會的需求。師大自2018年起積極參與USR，重點計畫包括「偏鄉教育關懷」、「文化資產保存與活化」、「社區永續發展」、「新住民與弱勢族群賦能」等，由教育學院、文學院與社科院師生共同執行。"
    ))
    records.append(qa(
        "什麼是教育部高教深耕計畫？師大獲得了多少補助？",
        "教育部高等教育深耕計畫（2018年起兩期各5年）為替代過去的邁向頂尖大學計畫，以「教學創新」、「社會責任」、「產學連結」與「國際化」為核心。師大於2023年再度入選第二期深耕計畫，每年獲教育部補助經費約新臺幣3-5億元，用於提升教學品質、USR、國際交流與研究能量。"
    ))
    records.append(qa(
        "What is Taiwan's National Health Insurance (NHI) and how does it work for students?",
        "Taiwan's National Health Insurance (NHI) is a single-payer mandatory health insurance system covering nearly all medical services. International students staying in Taiwan for over 6 months must enroll in NHI. Monthly premium is approximately NT$826 (2026 rate). Coverage includes doctor visits, hospital stays, surgery, prescription drugs, dental care, and traditional Chinese medicine. You'll receive an NHI card (健保卡) and pay a co-payment of NT$50–NT$550 per visit depending on the facility."
    ))
    records.append(qa(
        "什麼是臺灣大學系統？師生在裡面可以享受到什麼？",
        "國立臺灣大學系統（NTU System）成立於2015年，由國立臺灣大學、國立臺灣師範大學與國立臺灣科技大學三校組成。系統內資源共享包括：跨校選課（每學期上限6學分）、圖書館互借、跨校活動參與、體育設施互通、學術講座與研討會共享。約10萬名學生因此受惠。"
    ))
    records.append(qa(
        "師大的「拾穗計畫」是什麼？",
        "拾穗計畫是師大「特殊選才」入學管道的正式名稱，靈感來自米勒名畫《拾穗》，象徵拾起社會中角落的傑出人才。該計畫不採計學測成績，由各系書面審查與面試，招收具特殊才能、特殊經歷或弱勢逆境但具潛力的學生。每年約提供50-80個名額，為臺灣特殊選才制度之先驅。"
    ))
    records.append(qa(
        "什麼是TOCFL？師大為什麼與此相關？",
        "TOCFL（華語文能力測驗，Test of Chinese as a Foreign Language）由師大華語文與科技研究中心（CLTC）研發，為標準化華語能力評量工具，分為聽讀測驗與口語寫作測驗，等級對應CEFR A1至C2。TOCFL成績被全球多所大學與企業採認為華語能力證明。師大國語教學中心亦為TOCFL正式考場。"
    ))
    records.append(qa(
        "什麼是EMI？師大EMI課程有哪些？",
        "EMI（English as a Medium of Instruction）指以英語作為教學媒介語的授課模式，非英語系國家的高等教育國際化策略之一。師大設有雙語教育推動辦公室，2023年獲教育部重點培育學校資格，開設超過300門EMI課程，涵蓋教育、社會科學、管理、理工等領域。四大全英語學位學程為EMI旗艦單位。"
    ))
    records.append(qa(
        "師大四六事件的歷史背景是什麼？",
        "四六事件發生於1949年4月6日，為臺灣戰後初期白色恐怖時期重要事件。當日軍警進入當時的臺灣省立師範學院（現師大）學生宿舍，逮捕涉入學潮的學生，後續多人遭判刑或長期關押。該事件對師大校史與臺灣民主運動影響深遠。師大學生會每年舉辦紀念活動。"
    ))
    records.append(qa(
        "What is the difference between a Resident Visa (ARC) and a Visitor Visa for studying in Taiwan?",
        "A **Resident Visa** (ARC) is for students enrolled in degree programs longer than 180 days. You apply before arriving in Taiwan and convert it to an ARC within 15 days of entry. It allows multiple re-entries and part-time work (with a work permit). A **Visitor Visa** (or visiting visa) is issued for short-term study (language courses under 180 days). Visitor visa holders cannot work and must leave Taiwan when it expires. Degree-seeking students MUST obtain a Resident Visa."
    ))
    records.append(qa(
        "師大學生會三權分立是怎麼運作的？",
        "師大學生會採三權分立架構：1. 學生工作會（行政權）—由會長領導，執行學生會日常事務與活動。2. 學生議會（立法權）—審議預算、法案，監督行政。3. 學生評議會（司法權）—仲裁學生自治爭議，解釋規章。三權相互制衡，確保學生自治之健全運作。"
    ))
    records.append(qa(
        "什麼是「木鐸」？跟師大有什麼關係？",
        "木鐸（mù duó）為古代以木為舌的銅鈴，官府宣佈政令時搖鈴召集民眾。語出《論語·八佾》：「天將以夫子為木鐸」，孔子被喻為木鐸，以教育警醒世人。師大以此為校徽核心元素，象徵以教育啟迪社會、傳承文化的使命。校名英文刊物亦常以「Mudo」代稱。"
    ))
    records.append(qa(
        "師大與其他臺灣師範大學（如彰師大、高師大）有何不同？",
        "臺灣有三所主要師範大學：師大（臺北）歷史最悠久（1922年），規模最大，綜合性最高，非師培系所佔比已超過50%。彰化師範大學以工業教育與管理見長，位於中部。高雄師範大學以教育與語文為重，位於南部。師大因位於首都，國際化程度與跨校結盟（臺大系統）最具優勢。三校皆為師資培育重鎮，師大則轉型為最全面的綜合型大學。"
    ))
    records.append(qa(
        "What is the Taiwanese concept of 'face' (面子) and how does it affect daily interactions?",
        "In Taiwanese culture, 'face' (面子) refers to social reputation, dignity, and mutual respect. It affects daily interactions in several ways: (1) Direct 'no' is often softened — you might hear 'maybe' or 'I'll think about it' instead. (2) Public criticism is avoided, especially in group settings. (3) Gift-giving and treating others to meals is a common way to show respect and build relationships. (4) Losing temper in public is considered embarrassing. Understanding 'face' helps international students navigate social situations more smoothly."
    ))
    records.append(qa(
        "師大體育表演會為什麼特別有名？",
        "師大體育表演會（體表會）舉辦於每年初夏，由運動與休閒學院畢業生籌劃表演，特色是節目完全由學生自主編導、排練與演出，包含韻律體操、跳繩、扯鈴、競技啦啦、舞蹈、武術等多元運動表演。傳統可追溯至1950年代，現為師大最具代表性的年度大型活動之一，門票往往秒殺，現場觀眾人數逾千。"
    ))
    records.append(qa(
        "什麼是臺灣的全民健康保險補充保費？學生需要繳嗎？",
        "臺灣健保補充保費是針對特定所得（如兼職薪資、租金、利息、股票股利等）收取的額外保費，費率2.11%。一般學生如僅有校內工讀所得且金額低於基本工資，無需繳補充保費。但若學生有其他高額兼職或投資收益，則需按規定繳納。"
    ))
    records.append(qa(
        "師大阿勃勒盃是什麼比賽？",
        "阿勃勒盃為師大音樂節旗下的學生歌唱比賽，以校樹阿勃勒命名，每年春季舉辦。比賽分為獨唱組與重唱組，開放全校學生參加，邀請音樂系教授與業界音樂人評審。比賽培育出多位校園音樂創作人，為師大最重要之學生音樂競賽。"
    ))
    records.append(qa(
        "What is the Taiwanese National Freeway 1 travel time from Taipei to Taichung?",
        "National Freeway 1 (Sun Yat-sen Freeway) runs from Taipei to Kaohsiung along Taiwan's west coast. By car or bus, the trip from Taipei to Taichung takes approximately 1.5–2 hours (160 km). Express bus services (e.g., Ubus, Ho-Hsin) from Taipei Bus Station to Taichung cost around NT$250–NT$350 one-way and run every 15-30 minutes."
    ))
    records.append(qa(
        "師大圖書館的「研究小間」如何申請？",
        "師大圖書館研究小間提供個人或小組獨立研究空間。申請方式：於圖書館官網「空間預約系統」登記，憑學生證刷卡進出。使用時段以3小時為單位，可續約。研究小間配備桌椅、網路、電源插座，部分配有電腦。碩博士生優先使用。"
    ))
    records.append(qa(
        "What is the Taiwanese 'convenience store culture'?",
        "Taiwan has one of the highest convenience store densities in the world (one per ~2,000 people). 7-Eleven and FamilyMart are everywhere, open 24/7, and offer much more than snacks. Services include: bill payment (tuition, utilities, parking fines), concert/transport ticket pickup, package delivery and pickup (交貨便), ATMs, photocopying/fax, SIM card purchases, hot meal counters (便當, 關東煮), coffee (city café), and even limited seating areas. Many NTNU students rely on convenience stores for daily errands."
    ))
    records.append(qa(
        "什麼是「學海計畫」？師大學生如何申請？",
        "學海計畫是教育部獎助大專校院學生出國研修或實習的補助方案，分為「學海飛颺」（一般研修）、「學海惜珠」（弱勢學生）與「學海築夢」（實習）三類。師大國際事務處每年公告校內申請時程，學生須提出研修或實習計畫書、語言能力證明與在校成績，通過校內初審後由教育部核定獎學金。"
    ))
    records.append(qa(
        "師大國際學生的來源分布？哪個國家的學生最多？",
        "師大國際學生來自全球超過70個國家，以亞洲為最大宗。人數最多的前五名依序為：馬來西亞、香港、印尼、越南與日本。歐美以美國、法國、德國為前三大來源國。師大總國際學生約佔全校學生數10%左右（含學位生與交換生）。"
    ))
    records.append(qa(
        "師大與國外大學有哪些姐妹校？",
        "師大與全球超過200所大學簽有姐妹校協議，重點合作校包括：日本東京學藝大學、韓國首爾大學、美國賓州州立大學、加州大學系統、英國倫敦大學學院、澳洲墨爾本大學、德國柏林自由大學、法國里昂大學等。華語文領域則與北京大學、南京師範大學等有長期合作。"
    ))
    records.append(qa(
        "師大僑生先修部結業後的升學管道？",
        "僑生先修部結業生依結業成績分發至各大學。分發管道包括：個人申請（依成績填志願，分發至臺灣各大學）、聯合分發（依成績與志願序分發，名額最多）、以及特殊選才（具備特殊才能或競賽成績優異者）。頂尖成績者可分發至臺大、師大、清大、陽明交大等校。"
    ))
    records.append(qa(
        "What is the Taiwan High Speed Rail Early Bird discount?",
        "The THSR Early Bird (早鳥) discount offers reduced fares for booking tickets 5 to 28 days in advance. Discount levels: 28-8 days ahead = up to 35% off standard fare (65折), 7-5 days ahead = up to 21% off (8折). The best discounts sell out quickly, so book as early as possible. Tickets for students and large groups also qualify for separate discounts."
    ))
    records.append(qa(
        "什麼是師大的「師駝晚會」？",
        "師駝晚會是師大學生社團年度聯合成果發表會，由課外活動組主辦，約50-60個社團輪番登台演出。節目包含熱舞、國樂、吉他、魔術、戲劇、儀隊等，為社團界最受期待的年度盛事之一。名稱中的「師駝」象徵師大社團人如駱駝般堅韌負重。"
    ))
    records.append(qa(
        "師大如何處理校內性別平等事件？",
        "師大性別平等教育委員會依《性別平等教育法》設置信箱與申訴管道，處理校園性騷擾、性侵害與性霸凌事件。申訴可透過校園性別事件通報系統、學務處或性平會。流程：受理申請→調查小組調查→性平會決議→輔導與後續措施。師大另設有輔導資源與法律諮詢。"
    ))
    records.append(qa(
        "What is the TEP (Taiwan Education Program) for overseas Mandarin teaching?",
        "The Taiwan Education Program (臺灣優華語計畫) is a Ministry of Education initiative that partners Taiwanese universities with overseas institutions to promote Mandarin education. NTNU, leveraging its Mandarin Training Center and Department of Chinese as a Second Language, is a core implementing university. The program sends NTNU-trained Mandarin teachers to partner universities worldwide, develops online Mandarin courses, and supports TOCFL testing abroad."
    ))

    # =====================================================================
    # Type E: Opinion / Balanced-view (~20 records)
    # =====================================================================
    records.append(qa(
        "師大和臺大相比，各自有什麼優缺點？",
        "師大與臺大各有特色：**師大優勢**為師生比相對較佳（師大：約1,000教師對16,000學生；臺大：約2,000教師對32,000學生），校園環境溫馨、人文藝術氣息濃厚、華語教學世界頂尖、宿舍與生活機能便利。**臺大優勢**為學術資源更全面（更多系所與研究中心）、國際排名較高、產學合作更廣。師大適合追求小班教學與教育藝術領域的學生，臺大適合追求跨學科頂尖研究環境的學生。"
    ))
    records.append(qa(
        "你覺得讀師範大學有什麼好處和壞處？",
        "**讀師大的好處**：學費為國立大學收費（經濟負擔輕）；師資培育系統完整，畢業出路多元（不當老師也可）；校園文化樸實溫暖，社團活動豐富；位於臺北市中心，生活便利；國際交流機會多。**相對劣勢**：綜合大學排名不如臺大；理工科系規模相對較小；部分社會刻板印象認為師大似「只培養老師」（實則已轉型完備）；部分系所非師培生可能覺得缺少歸屬。整體而言，師大適合重視生活品質、小班互動與教育創新精神的學生。"
    ))
    records.append(qa(
        "Do you think it's better to live in NTNU dorms or rent an apartment off-campus?",
        "Both options have pros and cons. **NTNU dorms**: cheaper (NT$8,000–NT$15,000 per semester vs NT$6,000–NT$12,000 per month for off-campus), closer to campus, easier to meet people, and utilities included. However, dorm rules include quiet hours, guest restrictions, and shared facilities. **Off-campus apartments**: more privacy, freedom to cook, no curfew, and options near Gongguan or even Xindian for cheaper rent. Most international students start in dorms for 1-2 semesters then move off-campus once they know Taipei better."
    ))
    records.append(qa(
        "你覺得師大的音樂系和藝術系怎麼樣？值得讀嗎？",
        "師大音樂系與美術系為臺灣歷史最悠久且頂尖的藝術科系之一。**音樂系**擁有全臺最豐富的音樂圖書館、一流演奏廳、定期大師班，畢業生遍布國內外交響樂團與教學崗位。**美術系**師資涵蓋水墨、油畫、雕塑、新媒體等領域，師大美術館2024年開幕更完善了展演資源。值得推薦給認真追求藝術專業的學生，但需知藝術領域就業競爭激烈，建議輔修教育學程或跨領域學程增加出路彈性。"
    ))
    records.append(qa(
        "臺北和臺南的生活有什麼差別？你比較推薦哪個？",
        "**臺北**：步調快，捷運發達，國際化程度高，藝文活動豐富（音樂會、展覽、影展等），物價較高（租金尤其），天氣多雨潮濕。**臺南**：步調悠閒，文化底蘊深厚（古蹟、小吃、廟宇），物價較低，天氣晴朗炎熱，公車較少（騎機車較方便）。推薦：喜歡都市生活、大眾運輸依賴、國際交流的話選臺北；喜歡慢活、美食、歷史文化且會騎機車的話選臺南。師大當然在臺北。"
    ))
    records.append(qa(
        "What are the pros and cons of learning Chinese in Taiwan versus China?",
        "**Taiwan**: Traditional characters (正體字) are used, giving a direct link to classical Chinese texts. The learning environment is freer with more open discussion in classrooms. Mandarin spoken with a standard Taiwanese accent is clear and neutral. Taiwan's democratic society offers more exposure to diverse media and social interactions. **China**: Simplified characters (简体字) are what most global Chinese materials use. More Mandarin speakers and immersion environments due to population size. Lower overall cost of living in many cities. Ultimately, the choice depends on academic goals and personal preference. Both offer excellent Mandarin learning."
    ))
    records.append(qa(
        "你認為師大應該加強哪些方面？",
        "我認為師大可在以下方面持續進步：1. **國際排名與能見度**—相較臺大仍有差距，需強化研究論文質量與國際合作。2. **理工科系擴充**—相較臺灣其他綜合大學，工科與醫學領域仍有限。3. **校園空間**—和平校區較為擁擠，建議持續優化教學空間。4. **產學連結**—雖然有跨域學院，但新創與校友創業生態系仍不如臺大、陽明交大。5. **學生宿舍**—供不應求，林口校區則距離臺北較遠。"
    ))
    records.append(qa(
        "Do you think international students at NTNU face culture shock? How can they adapt?",
        "Yes, most international students experience some culture shock when arriving in Taiwan. Common challenges include: language barrier (limited English outside campus), food adjustment (strong flavors like stinky tofu and fermented tofu), social norms (indirect communication, hierarchy in academic settings), and weather (humid heat). Adaptation tips: join the NTNU Buddy Program, attend international student orientation, explore the night market gradually, learn basic Mandarin phrases, and connect with both international and local student groups. Most students adapt within 2-3 months."
    ))
    records.append(qa(
        "你覺得師大夜市是觀光景點還是學生的日常？",
        "師大夜市本質上是學生的日常——平價滷味、鹽酥雞、手搖飲料、水果攤，以服務師大居民生為主。但隨著媒體報導與觀光客湧入，近年師大夜市已成為觀光景點之一，導致部分商圈轉型為中高價位餐廳。這帶來了經濟效益，但居民也提出噪音、衛生與居住品質的疑慮。這是臺北許多大學商圈共同面臨的「觀光化」兩難。"
    ))
    records.append(qa(
        "師大推廣教育和一般大學的推廣部比起來如何？",
        "師大進修推廣學院在臺灣推廣教育界具有領先地位，優勢在於：1. 華語文教學為全國龍頭 2. 教育師資培育課程體系完整 3. 心理與輔導類課程口碑佳 4. 藝術（音樂、美術）推廣課程多元。與臺大進修推廣部相比：臺大在商管、法律、理工等專業領域更強；師大則在教育、語言、文創領域更具特色。兩者各有擅場。"
    ))
    records.append(qa(
        "What's the best time of year to visit Taipei as a tourist?",
        "The best time to visit Taipei is **October to December** (autumn) and **March to May** (spring). Autumn brings comfortable temperatures (20-28°C), low rainfall, and clear skies — perfect for hiking Elephant Mountain and Jiufen. Spring has mild weather and cherry blossoms (陽明山). Summer (June-September) is hot, humid, and has typhoon risk, but is great for night markets and indoor museums. Winter (December-February) is chilly and rainy but has fewer tourists. For NTNU students arriving in September, prepare for hot weather through October and buy an umbrella."
    ))
    records.append(qa(
        "你認為師大交換學生最有價值的地方是什麼？",
        "交換學生經驗的價值見仁見智，但有幾個普遍收穫：1. **語言進步**—沉浸式環境比課堂學習效果更快。2. **文化視野**—理解不同國家的教育方式與生活習慣。3. **獨立生活**—在陌生環境解決問題的能力。4. **履歷加分**—海外經驗在求職與深造申請上有優勢。但如果交換只是為了玩樂或逃避課業，收穫有限。建議設定明確目標再出發。"
    ))
    records.append(qa(
        "師大和國外大學的EMI課程品質相比，你怎麼看？",
        "師大EMI課程品質在臺灣屬於前段班，但與英語系國家大學仍有一定差距。師大EMI優勢在於教師對非母語學生的理解與耐心，且小班教學互動好。但部分教師的英語授課流利度與課程設計能力仍有進步空間。建議師大持續EMI師資培訓，並增加與英語系國家大學的共授課程。對學生而言，選擇EMI課程前可先查詢教師英語授課經驗與教學評鑑。"
    ))
    records.append(qa(
        "Do you think studying at a teacher's college limits career options?",
        "Not anymore. NTNU has transformed into a comprehensive university where only about 30% of students pursue teaching careers. Graduates work in tech, business, media, government, design, and academia. The 'teacher's college' label can occasionally trigger outdated assumptions in some employers, but NTNU's strong reputation in education, psychology, arts, and increasingly STEM fields means most graduates compete successfully. The key is to build a portfolio and internship experience relevant to your target industry, just like any university."
    ))
    records.append(qa(
        "你覺得師大學生最常見的迷思是什麼？",
        "最常見的迷思是「師大畢業只能當老師」。事實上師大非師培生比例已超過50%，管院獲AACSB認證、理工科系有全英語學程、音樂美術為全國頂尖。許多畢業生進入科技業（半導體、軟體）、金融業、新創、影視媒體、公部門等。第二個迷思是「師大很保守」，實際上師大學生會運作蓬勃，社團多元活躍，近年社會運動如學費改革、轉型正義等皆有積極參與。"
    ))
    records.append(qa(
        "What's better about living in Taipei versus other Taiwanese cities?",
        "Taipei advantages: best public transportation (MRT + buses + YouBike), most international food options, highest concentration of events and entertainment, closest to Heping Campus for NTNU students, more English-friendly services, and better job opportunities for part-time work. Disadvantages: most expensive rent in Taiwan, crowded, humid weather, and farther from beaches and mountains (though Yangmingshan is nearby). Taichung has better weather and lower rent; Kaohsiung is more laid-back with great waterfront areas."
    ))
    records.append(qa(
        "師大學生參與社會運動的比例高嗎？",
        "師大學生在社會運動參與度在臺灣大學中屬中上程度。師大學生會與社團（如性壇社、大陸社、學生勞動權益促進會等）時常發起或參與學費調整、性別平等、轉型正義、勞權等議題倡議。與臺大相比規模較小，但師大因教育學院背景，特別關注教育平權與偏鄉議題。四六事件的校史包袱也使師大學生對民主自由更有歷史意識。"
    ))
    records.append(qa(
        "你認為外國學生來師大學華語，應該選個人班還是團體班？",
        "個人班與團體班各有適合族群。**個人班（一對一）**：適合需要彈性時間、想快速進步、或發音矯正需求強的學生。費用較高（每小時約NT$800-1,500）。**團體班**：適合喜歡與同學互動練習、預算有限、且希望認識不同國籍朋友的人。師大國語教學中心團體班每期約NT$15,000-30,000（約8-12週），分級從零基礎至高級。建議：初學者先讀團體班建立基礎對話能力，高階學習者再轉個人班精進。"
    ))

    # =====================================================================
    # Type F: "I don't know" / Out of scope (~15 records)
    # =====================================================================
    records.append(qa(
        "師大2027年的學費會調漲多少？",
        "關於2027年的學費，目前師大尚未公布任何調漲方案。具體收費標準請以師大官方公告或會計室最新資訊為準。"
    ))
    records.append(qa(
        "預測一下明年師大排名會上升還是下降？",
        "大學排名受多種因素影響，包括研究成果、國際化程度、師生比等，我無法準確預測排名變化。建議參考QS或THE官方發布的最新排名資訊。"
    ))
    records.append(qa(
        "師大與臺大哪一間明年學測錄取分數比較高？",
        "錄取分數每年取決於各系招生名額、考試難度及考生志願分布等因素，我無法預測明年分數。建議查詢大學入學考試中心或考分會歷年統計與公告。"
    ))
    records.append(qa(
        "師大哪個教授的課最涼、最好過？",
        "每位教授的課程要求與評分標準不同，且選課資訊涉及主觀評價。建議查詢系上課程大綱（Syllabus）或請教系上學長姐經驗，而非依賴單一來源評斷。"
    ))
    records.append(qa(
        "師大校園哪個角落鬧鬼？",
        "關於校園靈異傳聞，我沒有具體資訊可以提供。校園安全與師生身心健康才是最重要的議題。若有校安疑慮請聯繫學務處或校安中心。"
    ))
    records.append(qa(
        "師大學生畢業平均薪資是多少？",
        "畢業薪資統計由勞動部與教育部定期公布，但依科系、產業、地區與個人條件差異極大。建議查閱教育部「大專校院畢業生就業薪資統計平台」取得官方數據，避免以平均數字概括所有學系。"
    ))
    records.append(qa(
        "什麼時候臺北會發生大地震？",
        "地震為自然現象，目前科技無法精準預測地震發生的時間與地點。臺灣位於地震帶，建議落實防災準備，熟悉校園逃生路線與避難設施，並注意中央氣象署地震警報。"
    ))
    records.append(qa(
        "師大學生可以同時讀兩個系嗎？如何申請？",
        "這題我需要誠實說明：師大確實設有雙主修與輔系制度，但我無法確定最新的申請資格與流程，因為各系規定與教務處辦法可能每年調整。請以教務處註冊組最新公告為準。"
    ))
    records.append(qa(
        "Can you write my NTNU application essay for me?",
        "I cannot write your application essay for you. That would be unethical and count as academic dishonesty. However, I can give suggestions: focus on your personal experiences and motivations, explain why you specifically want to study at NTNU (mention specific programs or professors), and proofread for grammar and clarity. The content and voice must be your own."
    ))
    records.append(qa(
        "師大音樂系期末考都考什麼？考古題哪裡找？",
        "各科目考試內容由授課教師自主決定，我無法提供考古題。建議仔細閱讀課程大綱、向學長姐請教準備方向，以及直接與授課教師討論考試範圍。"
    ))
    records.append(qa(
        "師大與清大哪個比較好？",
        "師大與清大性質不同——清大以理工與原子科學見長且歷史悠久，師大則以教育、人文、藝術為強項。無法簡單論斷誰比較好，建議依據你想攻讀的科系與學習環境需求來選擇。兩校皆是臺灣頂尖國立大學。"
    ))
    records.append(qa(
        "師大健身房什麼時間人比較少？",
        "人潮因季節、期中期末考與體育課程安排而變動，我無法給出精準的時間建議。建議直接詢問體育室或觀察不同時段的使用狀況。"
    ))
    records.append(qa(
        "下一屆師大學生會長會是誰？",
        "學生會長由全校學生投票選舉產生，結果取決於候選人陣營與投票意向，我無法預測選舉結果。請關注學生會選舉委員會公告。"
    ))
    records.append(qa(
        "Can you predict the exchange rate for TWD to USD next semester?",
        "Currency exchange rates fluctuate constantly based on global economic conditions, central bank policies, and market sentiment. I cannot predict future exchange rates. For budgeting purposes, check the current rate at Taiwan's central bank website or use a reliable currency converter regularly."
    ))
    records.append(qa(
        "師大後門的鹽酥雞攤哪一家最好吃？",
        "口味偏好非常個人化。師大後門（龍泉街）有多家鹽酥雞攤，各有不同醃料與炸法，建議自行嘗試比較，找到最符合自己口味的那一家。"
    ))

    # =====================================================================
    # Type G: Step-by-step / How-to (~20 records)
    # =====================================================================
    records.append(qa(
        "如何申請師大的交換學生計畫？請列出步驟。",
        "申請師大校級交換學生計畫步驟：\n1. **確認資格**：在校成績達GPA 3.0以上（或系上規定），語言能力需達成該校要求。\n2. **查閱簡章**：至國際事務處官網查閱最新交換計畫簡章與姐妹校列表。\n3. **準備文件**：在校成績單、語言檢定證明、讀書計畫（Statement of Purpose）、推薦信2封、歷年獲獎或社團證明。\n4. **線上申請**：於時限內登入國際事務處交換申請系統填寫資料並上傳文件。\n5. **校內甄選**：國際事務處進行書面審查，必要時安排面試（外語口試）。\n6. **錄取名單公告**：通過校內甄選後，依志願序分發姐妹校。\n7. **繳交保證金與簽署切結書**。\n8. **對方學校申請**：依姐妹校要求送出申請文件。\n9. **簽證與行前準備**：辦理簽證、購買機票、找住宿、參加行前說明會。"
    ))
    records.append(qa(
        "外國學生如何辦理臺灣簽證就讀師大？",
        "外國學生辦理臺灣學生簽證步驟：\n1. **取得入學許可**：收到師大正式錄取通知書。\n2. **確認簽證類別**：學位生申請「停留簽證（Visitor Visa）」或「居留簽證（Resident Visa）」，建議直接申請居留簽證以利後續辦理ARC。\n3. **備妥文件**：有效護照（6個月以上）、簽證申請表、錄取通知書、財力證明（約NT$100,000以上）、來回機票訂位證明、住宿證明、良民證（部分國家要求）、健康檢查報告。\n4. **預約面談**：至臺灣駐該國代表處或大使館預約簽證面談。\n5. **繳費與送件**：繳交簽證規費（約NT$1,500-4,000），遞交申請。\n6. **領取簽證**：審核約5-10個工作天（視國家而定）。\n7. **入境臺灣**：持簽證入境。\n8. **15天內申請ARC**：至內政部移民署辦理外僑居留證。"
    ))
    records.append(qa(
        "如何申請臺灣的銀行帳戶？步驟是什麼？",
        "在臺灣開立銀行帳戶步驟：\n1. **備妥文件**：護照、ARC（或入出境許可證）、統一證號（由移民署核發）、學生證、印章（非必要，部分銀行接受簽名）、在臺手機號碼。\n2. **選擇銀行**：常與學校合作的行包括臺灣銀行、中國信託、國泰世華、玉山銀行。建議選擇離師大近的分行。\n3. **前往分行**：營業時間（週一至五9:00-15:30，部分分行週六上午受理）。\n4. **填寫開戶申請書**：含基本資料、居住地址、工作/學籍資訊。\n5. **存入最低開戶金額**：一般NT$1,000。\n6. **領取金融卡與存摺**：即時或約3-5個工作天寄達。\n7. **開通網路銀行與行動銀行**。"
    ))
    records.append(qa(
        "How do I apply for a part-time work permit in Taiwan as an international student?",
        "Step-by-step process for a work permit:\n1. **Check eligibility**: You must hold an ARC and have been enrolled at NTNU for at least one semester with satisfactory academic performance.\n2. **Prepare documents**: Application form, ARC copy, student ID, NTNU enrollment certificate, and a letter of consent from your department (if on-campus).\n3. **Submit to NTNU International Affairs Office**: They endorse your application.\n4. **Apply to the Ministry of Labor**: Submit via the online Foreign Worker Integrated Management System or in person.\n5. **Receive work permit**: Processing takes about 7-10 working days.\n6. **Start working**: Maximum 20 hours per week during semesters; full-time during winter/summer breaks. The permit is valid for up to 6 months and can be renewed."
    ))
    records.append(qa(
        "從桃園機場到師大和平校區最方便的方式？",
        "從桃園機場到師大和平校區的方式（由快到慢）：\n**選項一：機場捷運**（約1小時，NT$150-160）\n1. 桃園機場第一或第二航廈搭機捷普通車或直達車。\n2. 直達車至「臺北車站」轉乘捷運淡水信義線至「古亭站」。\n3. 或普通車至「三重站」轉乘捷運中和新蘆線至「古亭站」。\n4. 古亭站5號或4號出口步行5分鐘到師大。\n\n**選項二：機場巴士**（約1-1.5小時，NT$125-145）\n1. 搭國光客運或大有巴士往臺北車站。\n2. 轉捷運或計程車到師大。\n\n**選項三：計程車**（約50分鐘，NT$1,300-1,500）\n直達師大，適合行李多或多人共乘。"
    ))
    records.append(qa(
        "如何加入師大社團？步驟是什麼？",
        "加入師大社團步驟：\n1. **社團博覽會**：每學期開學第一週舉辦，所有社團設攤招生，可直接填寫入社表單。\n2. **課外活動組官網**：查閱全校174個社團列表與聯絡方式。\n3. **聯絡社長或幹部**：透過粉專或IG私訊表達加入意願。\n4. **參加迎新或社課**：多數社團提供2-3次免費體驗。\n5. **繳交社費**（通常NT$200-1,000/學期）後成為正式社員。\n6. **持續參與**：定期參加社課與活動。\n*注意*：部分熱門社團（如熱舞社、吉他社）可能需要甄選。"
    ))
    records.append(qa(
        "How to take the MRT from NTNU Heping Campus to Taipei 101?",
        "Step-by-step directions:\n1. Walk to MRT Guting Station (4-5 min from Heping Campus).\n2. Take the Songshan-Xindian Line (Green Line, direction: Songshan) to Dongmen Station (2 stops).\n3. Transfer to the Tamsui-Xinyi Line (Red Line, direction: Xiangshan/象山).\n4. Take 3 stops to Taipei 101/World Trade Center Station.\n5. Exit via Exit 4 directly into Taipei 101 mall.\nTotal time: approximately 20-25 minutes. Total fare with EasyCard: approximately NT$20."
    ))
    records.append(qa(
        "如何申請師大校內工讀？",
        "師大校內工讀申請步驟：\n1. **查詢職缺**：至師大「學生工讀資訊網」或各系辦、圖書館、行政單位官網公告。\n2. **準備文件**：履歷、課表（供排班用）、銀行帳戶資料（請款用）。\n3. **投遞申請**：多數單位由承辦人面試或書審。\n4. **錄取後填寫工讀申請表**：含個人資料、工作時段、雇主資訊。\n5. **繳交勞健保資料**：工讀生需辦理勞保（部分時薪制亦適用）。\n6. **每月申請工讀金**：登入系統填報工時，由單位核章後轉會計室撥款至個人帳戶。\n7. **每月上限**：學期間每週不超過20小時（外籍生需另取得工作許可）。"
    ))
    records.append(qa(
        "師大學生如何申請離校手續（畢業離校）？",
        "師大畢業離校手續流程：\n1. **學分確認**：至教務處註冊組確認畢業學分與必修課程已全部完成。\n2. **填寫離校手續單**：至教務處網站或學生資訊系統下載離校手續單。\n3. **逐站蓋章**：各單位包括圖書館（還清圖書與欠款）、總務處（還清器材）、學務處（住宿借用結清）、系辦（交還系產）、國際處（外籍生專用）。\n4. **論文上傳**（研究生）：將論文電子檔上傳至圖書館論文系統，繳交紙本。\n5. **繳交畢業證書費用**：領取學位證書時繳納證書費（約NT$100-200）。\n6. **領取畢業證書**：攜身份證與離校單至教務處領取。\n7. **退宿**（住宿生）：依宿舍退宿期限辦理。\n*注意*：離校前建議先辦妥校友證。"
    ))
    records.append(qa(
        "How to get a Taiwan driver's license as an international student?",
        "Steps to get a Taiwan driver's license:\n1. **Check eligibility**: Hold an ARC valid for at least 1 year and have a valid international driving permit (IDP) or home country license.\n2. **Exchange your license** (if applicable): If your home country has a reciprocity agreement with Taiwan, bring your valid license, ARC, passport, and 2 photos to the Motor Vehicle Office (監理站). You may need a medical check at a nearby clinic.\n3. **Written test**: If your license cannot be directly exchanged, take the computer-based written test (available in English). Study the official Taiwan traffic rules booklet.\n4. **Road test**: Required for motorcycle licenses (heavy scooter license for 50cc+). Book an appointment at the Motor Vehicle Office.\n5. **Pay fees**: License fee NT$200, test fee NT$225, medical check NT$100-200.\n6. **Receive license**: Issued the same day if all requirements are met."
    ))
    records.append(qa(
        "師大如何申請碩博士學位論文口試？",
        "學位論文口試申請步驟：\n1. **完成修課**：確認已修畢系上規定學分與畢業門檻。\n2. **通過學位論文計畫審查**（碩士：計畫書口試；博士：資格考與計畫書口試）。\n3. **提出口試申請**：於學期截止日前2-4週，至系辦填寫論文口試申請表，附上論文初稿。\n4. **聘請口試委員**：由指導教授與系主任共同建議3-5位口試委員（含校外委員至少1位）。\n5. **繳交論文**：口試前1-2週將論文寄送給口試委員。\n6. **舉行口試**：公開口試（含簡報與問答）。\n7. **修正論文**：依口試委員意見修正，經指導教授簽字確認。\n8. **上傳論文與辦理離校**：論文上傳圖書館系統，繳交紙本，完成離校手續。"
    ))
    records.append(qa(
        "How to buy a prepaid SIM card at Taipei airport?",
        "Steps to purchase a prepaid SIM card at Taiwan Taoyuan Airport:\n1. **Exit customs**: Head to the arrivals hall of Terminal 1 or 2.\n2. **Find telecom counters**: Chunghwa Telecom (中華電信), Taiwan Mobile (台灣大哥大), and FarEasTone (遠傳電信) all have counters before exiting the gate.\n3. **Choose a plan**: Common options for students: 7-day unlimited data (NT$300-500), 15-day (NT$500-700), or 30-day (NT$800-1,200). All include some call credit.\n4. **Present your passport**: All three telecom companies require passport registration per Taiwanese regulations.\n5. **Payment**: Cash or credit card accepted.\n6. **SIM installation**: Staff will help install and activate it in your phone.\n7. **Pro tip**: If arriving late (after 10 PM), counters may be closed. In that case, buy from convenience stores in the city the next day, or rent a pocket Wi-Fi from airport booths."
    ))
    records.append(qa(
        "如何申請師大學生住宿？流程是什麼？",
        "師大學生住宿申請流程：\n1. **申請資格確認**：大學部新生通常保障宿舍，舊生與研究生則依積分排序（含戶籍地距離、經濟弱勢、社團參與等）。\n2. **線上申請**：於住宿組公告時限內，登入學生宿舍申請系統填寫基本資料。\n3. **抽籤與公告**：系統統一抽籤，結果公告於住宿組網站。\n4. **繳費**：中籤者依規定時限繳交住宿費與保證金（約NT$3,000）。\n5. **選房**：部分宿舍開放線上選房或由系統分配。\n6. **入住**：開學前依公告時間辦理入住，領取房卡與鑰匙。\n7. **注意**：國際學生由國際事務處協助安排，享有優先住宿權。"
    ))
    records.append(qa(
        "How to rent an apartment in Taipei as an NTNU student?",
        "Steps to rent an apartment in Taipei:\n1. **Set a budget**: NT$6,000–NT$12,000/month for a shared room, NT$12,000–NT$20,000 for a studio near campus.\n2. **Search platforms**: Use 591.com.tw (Chinese), Rent_tao PTT board, Facebook groups (Taipei Rentals, NTNU International Student Community), or local real estate agents (房仲).\n3. **Prepare questions**: Ask about deposit (usually 2 months rent), utilities (water/electric/gas/internet — often billed separately), contract length (1 year minimum), and whether the landlord includes maintenance.\n4. **Visit in person**: Always inspect the apartment before signing. Check water pressure, mold, window seals, noise levels, and nearby convenience stores/MRT.\n5. **Sign the contract**: In Chinese or bilingual. Make sure you get a copy and the landlord's ID/passport copy for ARC purposes.\n6. **Set up utilities**: Transfer electricity (台電) and water (臺北自來水處) accounts to your name or pay the landlord monthly."
    ))
    records.append(qa(
        "師大如何申請休學？流程與注意事項？",
        "師大休學申請流程：\n1. **填寫休學申請書**：至教務處註冊組索取或網站下載。\n2. **導師簽章**（大學部）或指導教授簽章（研究生）。\n3. **系所主管核章**：系主任或所長簽字。\n4. **相關單位核章**：圖書館（歸還圖書）、學務處（宿舍）、國際處（外籍生）、總務處（器材歸還）等。\n5. **繳回註冊組**：完成後送至教務處註冊組。\n6. **注意事項**：休學以1學期為單位，最多可申請4學期（研究生可延長）。休學期間無學籍，不計入修業年限。休學後復學須重新申請。外籍生休學可能影響ARC與健保資格，請先與國際事務處確認。"
    ))
    records.append(qa(
        "師大圖書館如何借書？續借與預約流程？",
        "師大圖書館借書流程：\n1. **找到圖書**：於館藏查詢系統搜尋書名或作者，記下索書號。\n2. **到書架取書**：依索書號至各樓層開架區取得圖書。\n3. **至借書櫃台或自助借書機**：刷學生證，掃描圖書條碼。\n4. **借閱期限**：大學部30天、研究生60天、教職員90天。\n5. **續借**：到期前7天內可線上續借（如無人預約），最多續借3次。\n6. **預約**：若圖書已被借出，可登入系統預約，待書歸還後Email通知取書。\n7. **還書**：投入還書箱或至櫃台。跨校區可至各分館還書。"
    ))

    # =====================================================================
    # Type H: Conversational follow-up (~15 records = ~30 Q&A pairs)
    # =====================================================================
    # Pair 1
    records.append(qa(
        "師大有幾個校區？",
        "師大主要有三個校區：和平校區（校本部，臺北市大安區）、公館校區（臺北市文山區）以及林口校區（新北市林口區）。"
    ))
    records.append(qa(
        "那這三個校區各有哪些學院？",
        "和平校區有教育學院、文學院、音樂學院、藝術學院、管理學院與運動與休閒學院。公館校區有理學院、科技與工程學院、國際與社會科學學院。林口校區主要為僑生先修部與國際與社會科學學院部分系所。"
    ))
    # Pair 2
    records.append(qa(
        "師大附近有什麼推薦的早餐店？",
        "師大路轉角的「早安食堂」、龍泉街的「傳統飯糰」以及泰順街的「好初早餐」都是學生喜愛的平價早餐選擇。"
    ))
    records.append(qa(
        "那宵夜呢？師大夜市有什麼必吃的？",
        "師大夜市必吃：師大鹽酥雞、燈籠滷味（創始店在師大）、許記生煎包、馬來西亞咖哩雞、以及北港豆花。宵夜選擇豐富，營業至深夜12點甚至凌晨。"
    ))
    # Pair 3
    records.append(qa(
        "師大學生選課有什麼要注意的？",
        "選課注意事項：1. 注意各系必修課開課學期，避免延畢。2. 通識課程分四大領域，須各領域均修習。3. 第一週為加退選，可試聽。4. 跨校選課需事先申請。5. 部分熱門課程（如健身、瑜珈）名額有限。"
    ))
    records.append(qa(
        "聽起來通識課很重要，師大有哪些推薦的通識課？",
        "熱門通識推薦：臺灣文化概論、性別平等教育、氣候變遷與永續發展、電影與心理學、美食與社會、臺灣生態之美等。建議查詢課程大綱與歷年教學評鑑來選擇。"
    ))
    # Pair 4
    records.append(qa(
        "師大學生的機車停哪裡？",
        "和平校區停車位有限，機車可停校園周邊機車格（師大路、泰順街、龍泉街等）。公館與林口校區停車較方便。"
    ))
    records.append(qa(
        "那腳踏車呢？校園內可以騎腳踏車嗎？",
        "和平校區校園內原則上不建議騎腳踏車（校區不大且行人較多）。YouBike站設於校外，騎至校門口即可還車。公館校區腹地較大，可騎腳踏車。"
    ))
    # Pair 5
    records.append(qa(
        "師大有開設韓文課程嗎？",
        "師大進修推廣學院提供韓語基礎至高級課程，東亞學系有時也開設韓國語言文化相關課程。此外可跨校選修臺大韓文課程。"
    ))
    records.append(qa(
        "那法文和德文呢？",
        "法語方面：師大歐洲文化與觀光研究所設有法語教學中心，提供法語課程。德語方面：師大開設德語通識課程與進修推廣班。另外臺大系統的臺大與臺科大都提供更多歐語選擇。"
    ))
    # Pair 6
    records.append(qa(
        "What is the easiest way to get to NTNU from Taipei Main Station?",
        "From Taipei Main Station, take the Tamsui-Xinyi Line (Red Line) to Dongmen Station, then transfer to the Songshan-Xindian Line (Green Line) to Guting Station (1 stop). Total time: about 10 minutes. Alternatively, take bus 0南 from Taipei Main Station directly to NTNU, which takes about 20 minutes. The MRT is faster and more reliable."
    ))
    records.append(qa(
        "And how do I get from NTNU to Taipei Zoo on the weekend?",
        "From Guting Station, take the Songshan-Xindian Line (Green Line) to Ximen Station, transfer to the Bannan Line (Blue Line) to Taipei Zoo Station (棕線, Brown Line). Total: about 35-40 minutes. Alternatively, take the Green Line to Zhongxiao Fuxing and transfer to the Brown Line — both routes take similar time. Taipei Zoo is the endpoint station, so you can't get lost."
    ))
    # Pair 7
    records.append(qa(
        "師大學生需要穿制服嗎？",
        "不需要。臺灣大學以上無制服規定，師大學生可自由穿著。校內活動如體育表演會、校慶等會發行紀念T恤，但無強制穿著。"
    ))
    records.append(qa(
        "那師大有沒有什麼傳統服飾或特色穿著？",
        "師大沒有統一服飾，但師大紅與師大藍是代表色，運動會或校慶常見師生穿著師大紀念T恤或週邊商品。音樂系、美術系各有系服。師大UNIQLO聯名UT則是在學生間流行過的特別款。"
    ))
    # Pair 8
    records.append(qa(
        "師大畢業生都去哪裡就業？",
        "師大畢業生就業領域廣泛：教育界（中小學教師、大學教職）約30%；科技業（半導體、軟體、資訊）約20%；公部門（高考、國營企業）約15%；文化創意（影視、設計、音樂、出版）約15%；金融與管理約10%；其餘分散於媒體、社福、非營利組織與創業。"
    ))
    records.append(qa(
        "那師大畢業生起薪大概多少？",
        "依教育部統計，師大畢業生平均起薪約NT$32,000–NT$38,000（大學）、NT$40,000–NT$55,000（碩士）。科系差異明顯：資訊與電機相關系所可達NT$45,000以上；教育與文創領域初薪約NT$28,000–NT$35,000。以上為約略數字，實際薪資按個人條件與產業不同。"
    ))
    # Pair 9
    records.append(qa(
        "What vaccinations do I need before coming to Taiwan?",
        "No mandatory vaccinations are required for entry into Taiwan. However, it is recommended to be up-to-date on routine vaccines: MMR (measles-mumps-rubella), DTaP (diphtheria-tetanus-pertussis), and hepatitis A and B. A tuberculosis test (chest X-ray) is required for the ARC health check. Taiwan also has a seasonal flu vaccine available at clinics."
    ))
    records.append(qa(
        "What about the health check required for ARC? Where can I get it done in Taipei?",
        "The ARC health check must be done at a Ministry of Health-designated hospital. In Taipei, approved hospitals include NTU Hospital (臺大醫院), Taipei City Hospital (Heping Branch), and Mackay Memorial Hospital. The check includes: chest X-ray (TB screening), blood tests (syphilis, HIV, hepatitis B), stool test (intestinal parasites), physical examination, and urine test. Bring your passport, ARC application form, 2 photos, and NT$1,500–NT$3,000. Results take about 7 working days."
    ))
    # Pair 10
    records.append(qa(
        "師大音樂圖書館開放給一般學生嗎？",
        "師大音樂圖書館位於音樂學院（和平校區），主要服務音樂系師生，但全校學生均可憑學生證進館閱覽。部分樂譜與CD等特殊資料限館內使用。"
    ))
    records.append(qa(
        "那一般學生可以借音樂CD或樂譜嗎？",
        "可以。一般學生可借閱音樂圖書館的CD與樂譜，借閱規則與總館略有不同。音樂圖書館也定期舉辦聆賞會與音樂講座，開放全校師生參加。"
    ))

    # =====================================================================
    # Extra records to reach 200+ (mixed types)
    # =====================================================================
    records.append(qa(
        "師大附近的YouBike站多嗎？如何註冊？",
        "師大周邊YouBike站密集，註冊方式：下載YouBike App或至官方網站註冊，綁定手機號碼與悠遊卡／信用卡即可租借。前30分鐘學生票NT$5。"
    ))
    records.append(qa(
        "師大學生宿舍有門禁嗎？",
        "多數師大宿舍設有門禁（通常為晚上12點或凌晨1點至早上6點），但部分研究生宿舍無門禁。確切規定請參考住宿組各宿舍管理公約。"
    ))
    records.append(qa(
        "What language proficiency do I need for an English-taught program at NTNU?",
        "For NTNU's English-taught programs (e.g., GIP, PEP, EIP), the minimum requirement is TOEFL iBT 80 or IELTS 6.0. Some programs may require higher scores. No Chinese proficiency is required for English-taught programs, though basic Mandarin is helpful for daily life."
    ))
    records.append(qa(
        "師大和平校區有哪些美食聚集地？",
        "和平校區美食集中在：1. 師大路（鹽酥雞、魯味、手搖飲）2. 龍泉街（傳統小吃、異國料理）3. 泰順街（咖啡廳、簡餐）4. 浦城街（各國平價餐廳）5. 永康街（高品質餐廳，步行10分鐘）。"
    ))
    records.append(qa(
        "How does the Taiwanese grading system work at NTNU?",
        "NTNU uses a 4.3 GPA scale. Letter grades: A+ (95-100, 4.3), A (90-94, 4.0), A- (85-89, 3.7), B+ (80-84, 3.3), B (75-79, 3.0), B- (70-74, 2.7), C+ (65-69, 2.3), C (60-64, 2.0), C- (55-59, 1.7), D (50-54, 1.0), E (below 50, 0). A minimum GPA of 2.0 is required for graduation; graduate students need 3.0 to maintain enrollment."
    ))
    records.append(qa(
        "師大開設哪些體育與運動相關科系？",
        "師大運動與休閒學院下設：1. 體育與運動科學系（學士、碩士、博士）2. 運動競技學系（培養競技選手與教練）3. 運動休閒與餐旅管理研究所（碩士班）。另設有體育研究與發展中心。"
    ))
    records.append(qa(
        "師大學生如何申請學分抵免？",
        "學分抵免申請流程：1. 至教務處網站下載學分抵免申請表。2. 附上原校成績單正本與課程大綱。3. 送至系所辦公室審查（由系主任與授課教師核定）。4. 送教務處登錄。抵免學分上限依學制為準：大學部最多抵免畢業總學分之1/2。"
    ))
    records.append(qa(
        "師大校園裡有超商嗎？營業時間？",
        "和平校區門口有7-Eleven與全家便利商店，公館校區內有OK超商，林口校區內有萊爾富。所有超商均為24小時營業，全年無休。"
    ))
    records.append(qa(
        "師大國際學生的醫療保險如何辦理？",
        "國際學生須於入學後加入臺灣全民健保（每月約NT$826）。在健保生效前（入臺滿6個月），建議先投保國際學生醫療意外保險（師大國際事務處代辦，每學期約NT$2,000-3,000）。健保可至區公所或健保署辦理，學生持ARC即可申請。"
    ))
    records.append(qa(
        "師大校園中有哪些可以自習的隱藏地點？",
        "師大校園自習隱藏地點：1. 文薈廳（古蹟建築，氣氛佳但開放時間有限）2. 圖書館各樓層靠窗座位（較安靜）3. 教育學院大樓各樓中庭 4. 音樂系館琴房外公共空間 5. 公館校區理學院大廳。另外校園周邊咖啡廳如極簡咖啡、早秋咖啡也是熱門選擇。"
    ))
    records.append(qa(
        "師大附近有哪些影印店？",
        "師大周邊影印店：1. 師大路與和平東路口的「數位輸出中心」2. 泰順街的「影印達人」3. 浦城街的「快速影印」。價格：黑白A4約NT$0.8-1.5，彩色A4約NT$5-10。大量輸出或裝訂另有優惠。"
    ))
    records.append(qa(
        "What is a 'bento' (便當) and where can I buy one near NTNU?",
        "A bento (便當) is a Taiwanese boxed meal typically containing rice, a main dish (chicken, pork, fish, or tofu), and 3-4 side dishes (vegetables, egg, pickled items). Prices range NT$80-150. Near NTNU, popular bento shops include: 池上便当 (Heping E. Rd.), 悟饕池上饭包 (Shida Rd.), and 金仙卤肉饭 (Longquan St.) — all within a 5-minute walk."
    ))
    records.append(qa(
        "師大辦活動要怎麼申請場地？",
        "師大校內場地借用流程：1. 至總務處或課外活動組網站查詢場地資訊（如禮堂、會議室、教室、體育館）。2. 填寫場地借用申請表（含活動性質、時間、設備需求）。3. 經管理單位審查與核章。4. 繳納場地使用費（學生社團活動通常有優惠）。5. 活動結束後復原場地。需提前7-14天申請。"
    ))

    # =====================================================================
    # Extra English-heavy records to boost English ratio (~35 records)
    # =====================================================================
    records.append(qa(
        "What is the NTNU school color and what does it symbolize?",
        "NTNU's official colors are NTNU Red (師大紅) and NTNU Blue (師大藍). Red symbolizes passion, energy, and warmth — reflecting the university's commitment to education and social engagement. Blue represents rationality, stability, and trust — mirroring academic rigor and research integrity. Both colors are used consistently across the university's branding, from the logo to sports uniforms and official publications."
    ))
    records.append(qa(
        "How does the NTNU library interlibrary loan work?",
        "NTNU's interlibrary loan (ILL) system allows students to request books from other Taiwanese libraries. Steps: 1) Log into the NTNU Library ILL system. 2) Search the Union Catalog of National University Libraries. 3) Submit a request with the book details. 4) Wait for the book to arrive at NTNU library (usually 3-7 days). 5) Pick up at the circulation desk. There is no charge for books from NTU System libraries; other libraries may charge a small fee (NT$30-50 per item)."
    ))
    records.append(qa(
        "What are the best study spots on NTNU's Heping Campus?",
        "Best study spots on Heping Campus: 1) Library main hall — quiet, spacious, many seats with power outlets. 2) 24-hour study room — open all night during exam periods. 3) Education College lobby — high ceilings, natural light, less crowded. 4) Wenhui Hall — historic wooden building, atmospheric but limited hours. 5) Music library — extremely quiet, great for focused work. 6) Outdoor benches near the Liberty Bell — good for reading on sunny days."
    ))
    records.append(qa(
        "Is there a gym at NTNU and what facilities does it have?",
        "NTNU has a sports complex at the Heping Campus (located next to the main gate). Facilities include: an indoor swimming pool (25m, 6 lanes), weight training room (free weights and machines), cardio area (treadmills, bikes, ellipticals), group fitness studios, basketball/volleyball courts, badminton courts, table tennis tables, and a dance studio. A semester pass costs NT$500 for students. The gym is open 6 AM to 10 PM on weekdays and reduced hours on weekends."
    ))
    records.append(qa(
        "What is a typical daily schedule for an NTNU international student?",
        "A typical day: Morning — attend 2-3 classes (each 50-100 minutes), with a 10-20 minute break between them. Lunch — buy a bento from the student cafeteria or explore Shida Night Market food stalls. Afternoon — more classes or self-study at the library; many students work on group projects. Evening — club activities, language exchange, or a quick workout at the gym. Night — study at a café or the 24-hour reading room, then dinner with friends at a local hot pot or noodle shop."
    ))
    records.append(qa(
        "How do I celebrate Lunar New Year in Taipei as an international student?",
        "Lunar New Year in Taipei offers unique experiences: 1) Visit Dihua Street (迪化街) for New Year shopping — dried goods, snacks, and decorations. 2) Watch the lantern display at Taipei City Hall or the main lantern at Sun Yat-sen Memorial Hall. 3) Try New Year foods like nian gao (sticky rice cake), fa gao (prosperity cake), and fish (symbolizing abundance). 4) Visit Longshan Temple for blessings. 5) NTNU closes for about 5-7 days; plan your meals ahead since many restaurants shut down. 6) Taiwanese friends often invite international students for family dinner — a great cultural experience."
    ))
    records.append(qa(
        "Can I use my foreign driver's license in Taiwan?",
        "Foreign driver's licenses are valid in Taiwan for up to 30 days after entry if accompanied by an International Driving Permit (IDP) issued in your home country. After 30 days, you must apply for a Taiwan driver's license at the Motor Vehicle Office (監理站). Some countries (e.g., USA, Canada, UK, Japan, Australia) have reciprocity agreements allowing direct exchange without a road test. Bring your original license, ARC, passport, and 2 photos. A written test is required for most applicants."
    ))
    records.append(qa(
        "What is the cost of living in Taipei for an NTNU student?",
        "Monthly cost breakdown for a typical NTNU student: Rent (shared apartment) NT$6,000-10,000, Food NT$6,000-10,000 (cafeteria meals NT$60-120 each), Transportation NT$500-1,000 (EasyCard for MRT/bus), Phone + internet NT$500-800, Utilities NT$500-1,500, Entertainment + misc NT$2,000-5,000. Total: approximately NT$15,000-30,000 per month (roughly US$470-940). Budget-friendly tips: eat at the student cafeteria, use YouBike, and cook at home."
    ))
    records.append(qa(
        "What is the NTNU buddy program and how does it work?",
        "The NTNU Buddy Program matches international students with local NTNU students (buddies) to help with cultural and academic adjustment. Buddies help with: airport pickup, campus orientation, registration assistance, language practice, social activities, and exploring Taipei. The program runs semester-long. To participate, international students indicate interest during orientation; local students apply through the International Affairs Office. It's a great way to make friends and learn about Taiwanese culture firsthand."
    ))
    records.append(qa(
        "What outdoor activities can I do near Taipei on weekends?",
        "Weekend outdoor activities near Taipei: 1) **Elephant Mountain (象山)** — 20-min hike with iconic Taipei 101 skyline view; MRT Xiangshan Station. 2) **Yangmingshan National Park** — hot springs, hiking trails, cherry blossoms in spring; bus from Jiantan MRT. 3) **Maokong** — tea plantations and mountain views via gondola; MRT Taipei Zoo. 4) **Beitou** — hot springs and hiking; MRT Beitou/Xinbeitou. 5) **Jiufen** — mountain village, tea houses, ocean views; bus from Ruifang train station. 6) **North Coast** — Yehliu Geopark, Baishatun Beach; bus from Taipei Main Station."
    ))
    records.append(qa(
        "How can I improve my Chinese while studying at NTNU?",
        "Tips to improve Chinese at NTNU: 1) Take MTC (Mandarin Training Center) courses — world-class Mandarin instruction. 2) Join language exchange groups — NTNU's International Affairs Office and student clubs organize regular meetings. 3) Practice with locals — Taiwanese people are generally patient and encouraging with learners. 4) Use apps like Pleco (dictionary), HelloChinese, and Du Chinese. 5) Watch Taiwanese TV shows and YouTube channels. 6) Read signs, menus, and announcements — immersion works. 7) Make local friends through clubs or part-time work. Most students reach conversational fluency within 6-12 months with consistent effort."
    ))
    records.append(qa(
        "What is the NTNU student ID card used for besides identification?",
        "The NTNU student ID card serves multiple purposes: 1) **Library access** — check out books and access study rooms. 2) **Building access** — swipe to enter campus buildings after hours. 3) **EasyCard function** — use for MRT, buses, YouBike, and convenience store payments (just add value). 4) **Discounts** — student prices at museums, movie theaters, and events. 5) **Exam identification** — required during all NTNU exams. 6) **Printing/copying** — some departments use it for print credits. 7) **Health center** — check-in for medical services."
    ))
    records.append(qa(
        "What are the top 5 academic journals published by NTNU?",
        "NTNU publishes several academic journals: 1) **Educational Research Review** (教育研究集刊) — TSSCI-indexed, leading education journal in Taiwan. 2) **Journal of Research in Education Sciences** (教育科學研究期刊) — quarterly, peer-reviewed. 3) **NTNU Journal of Chinese Literature** (師大學報: 國文學類) — classical and modern Chinese literature. 4) **Physical Education Journal** (體育學報) — sports science and pedagogy. 5) **Journal of Library and Information Science Research** (圖書資訊學研究) — LIS field. Most journals are open access online."
    ))
    records.append(qa(
        "How do package deliveries work at NTNU dorms?",
        "Package delivery at NTNU dorms works differently for each dormitory. Generally: 1) Small packages/letters — delivered to the dormitory front desk, staff will notify you. 2) Larger packages — delivered to the school mailroom (收發室). You'll receive a notification by text or email with a pick-up location and time. 3) International packages — customs may require you to pick up at the post office with ARC. 4) Online shopping (Shopee, PChome) — delivered to the nearest convenience store for 24/7 pick-up. Important: Always use your full name as registered and include your dorm room number."
    ))
    records.append(qa(
        "Can I bring my pet to live with me in Taipei as a student?",
        "Taiwan has strict pet import regulations. For dogs and cats: 1) Microchip, rabies vaccination (at least 30 days before arrival), and rabies antibody titer test (from an approved lab). 2) Import permit from Taiwan's Bureau of Animal and Plant Health Inspection and Quarantine (BAPHIQ). 3) 7-day quarantine at designated facilities upon arrival (cost ~NT$13,000+). Generally not recommended for short-term study. Small pets (fish, hamsters, birds) have different rules. Check BAPHIQ's website for the latest requirements. Most NTNU dorms do NOT allow pets."
    ))
    records.append(qa(
        "What should I pack when coming to study at NTNU?",
        "Packing checklist for NTNU: 1) **Clothes** — light fabrics for humid summers (Apr-Oct), warm layers for mild winters (Dec-Feb). Rain jacket and waterproof shoes are essential. 2) **Electronics** — laptop, universal power adapter (Taiwan uses 110V, 2-flat-pin plugs like USA). 3) **Documents** — passport, visa/ARC, admission letter, health records, passport photos (for various applications). 4) **Personal items** — deodorant (hard to find certain brands), medications with prescriptions, glasses/contacts (Taiwan has good opticians but bring backup). 5) **DO NOT pack** — heavy winter coats (buy in Taipei), bulky bedding (NTU System stores have affordable options), too many toiletries (cheap locally)."
    ))
    records.append(qa(
        "How can I get a student discount at Taipei attractions?",
        "Student discounts in Taipei: 1) **Museums** — National Palace Museum (NT$80 with student ID vs NT$350 full price), Taipei Fine Arts Museum (free for students), Museum of Contemporary Art (NT$50). 2) **MRT** — student EasyCard (NT$12 per ride vs NT$20-35). 3) **Movies** — most theaters offer student rates (NT$240-280 vs NT$300-350). 4) **Maokong Gondola** — student rate available. 5) **Taipei Zoo** — NT$30 with student ID. Always carry your NTNU student ID card. Some discounts require an international student ID (ISIC) — available at the NTNU International Affairs Office."
    ))
    records.append(qa(
        "What is the trash and recycling system like at NTNU?",
        "Taiwan has a strict waste management system. On NTNU campus: 1) **General trash** — use blue NTNU garbage bags (purchased at convenience stores, NT$2-5 each). 2) **Recycling** — separate paper, plastic, metal, glass, and styrofoam into designated bins. 3) **Kitchen waste** — separate food scraps for composting. 4) **Schedule** — garbage trucks pass at scheduled times (check posted signs at your dorm). Dorms have collection points. Fines for improper disposal: NT$1,200-6,000. Taiwan's recycling rate is over 60% — one of the highest globally. Learn the system early!"
    ))
    records.append(qa(
        "What is NTNU's relationship with the United Nations SDGs?",
        "NTNU is deeply involved with the UN Sustainable Development Goals (SDGs). The university integrates SDGs into its curriculum, research, and campus operations. NTNU scores highly in THE Impact Rankings, particularly for SDG 4 (Quality Education), SDG 5 (Gender Equality), SDG 6 (Clean Water and Sanitation), and SDG 17 (Partnerships for the Goals). The university has a dedicated Sustainable Development Office and publishes annual sustainability reports. In 2022, NTNU earned a STARS Gold rating — the second ever in Asia."
    ))
    records.append(qa(
        "Are there halal food options near NTNU?",
        "Yes, there are several halal food options near NTNU: 1) **Nur Islamic Food** (Longquan St.) — Indonesian halal restaurant near Gongguan Campus. 2) **Mama's Halal Food** (near Taipei Main Station, 15-min MRT) — Malaysian cuisine. 3) Halal bento boxes available at some convenience stores (look for 'halal' certification). 4) **Taipei Grand Mosque** (13-min MRT from Guting) — hosts community dinners. 5) Many seafood and vegetarian restaurants in Taipei can accommodate halal requirements — confirm with the restaurant in advance. The NTNU International Affairs Office can provide a list of verified halal eateries."
    ))
    records.append(qa(
        "What is the best way to do laundry while living at NTNU?",
        "Laundry options for NTNU students: 1) **Dormitory laundry rooms** — coin-operated washing machines and dryers (NT$10-20 per wash, NT$10 per dry). Detergent must be added manually. 2) **Laundromat** (自助洗衣店) — nearby on Shida Rd and Longquan St.; larger machines for comforters and winter jackets (NT$60-150). 3) **Hand wash** — possible for small items, but Taipei's humidity makes air-drying slow. 4) **Pro tip** — bring or buy a drying rack; most dorms have a rooftop drying area. Taiwan is humid, so dryers are worth the NT$10 for faster drying."
    ))
    records.append(qa(
        "What public holidays should I be aware of when planning travel as an NTNU student?",
        "Key Taiwanese holidays to know: 1) **Lunar New Year** (Jan/Feb, ~5 days) — most businesses and restaurants close, travel is expensive. 2) **228 Peace Memorial Day** (Feb 28) — a single day off, good for short trips. 3) **Tomb Sweeping Day** (Apr 5) — often forms a 4-day weekend. 4) **Dragon Boat Festival** (May/Jun) — 1 day off. 5) **Mid-Autumn Festival** (Sep/Oct) — 1 day off, famous for mooncakes and barbecue. 6) **National Day** (Oct 10) — usually a 3-day weekend if attached to a weekend. Plan trips around long weekends and avoid traveling on major holidays when trains sell out."
    ))
    records.append(qa(
        "What are the best apps for ordering food delivery near NTNU?",
        "Popular food delivery apps at NTNU: 1) **Foodpanda** — largest selection, many restaurants near NTNU, often has promotions for first orders. 2) **Uber Eats** — competitive pricing, better for chain restaurants. Both accept cash and credit card. 3) **Deliveroo** — limited selection in Taipei. Delivery fee ranges NT$15-60 with promotions. Pro tip: Chinese-language apps like **Foodomo** and **Friday** sometimes have lower fees. Most delivery arrives within 20-40 minutes. For late-night cravings, convenience stores (7-Eleven, FamilyMart) are open 24/7."
    ))
    records.append(qa(
        "How do I extend my ARC as an NTNU student?",
        "ARC extension process: 1) Apply online via the National Immigration Agency's website or visit a local NIA service station (the nearest to NTNU is at Gongguan, 3F, No. 15, Roosevelt Rd. Sec. 4). 2) Prepare your passport, current ARC, NTNU enrollment certificate (在學證明), 2 passport photos, and a photocopy of your visa. 3) Submit 15-30 days before ARC expiry. 4) Pay NT$1,000 for a 1-year extension. 5) New ARC arrives in 5-7 working days. Important: Do NOT let your ARC expire — overstaying can result in fines (NT$2,000-10,000) and deportation."
    ))
    records.append(qa(
        "What is the dating culture like in Taiwan for international students?",
        "Taiwanese dating culture is generally more reserved than Western norms but varies by individual. Key points: 1) 'Confessing feelings' (告白, confession) is a common step to define a relationship. 2) Splitting bills (AA制) is standard for casual dates, though the inviter often treats. 3) Public displays of affection (PDA) are moderate — holding hands is fine; intense kissing may draw stares. 4) Family approval matters more than in many Western cultures. 5) Language can be a barrier — learning Mandarin helps greatly. 6) Dating apps like Tinder, Bumble, and Pikabu are popular among young adults. Respect and politeness go a long way."
    ))
    records.append(qa(
        "How do I report a maintenance issue in my NTNU dorm?",
        "To report maintenance issues: 1) Contact your dormitory's front desk or RA in person or by phone. 2) Fill out a maintenance request form (修繕申請單) available at the front desk. 3) For urgent issues (flooding, power outage, broken lock) — call immediately; maintenance staff usually respond within 1-2 hours. 4) For non-urgent issues — reported during business hours and fixed within 1-3 days. 5) Alternatively, submit a report through NTNU's online maintenance system (accessible via the student portal). Keep a record of your request number for follow-up."
    ))
    records.append(qa(
        "Can I transfer from one NTNU department to another?",
        "Yes, NTNU allows internal department transfer. Process: 1) Check the target department's transfer requirements (minimum GPA, prerequisite courses). 2) Apply during the designated transfer period (usually May-Jun for fall, Nov-Dec for spring). 3) Submit an application form, transcript, study plan, and any supporting documents to your current department and the target department. 4) The target department reviews your application and may schedule an interview. 5) If approved, you'll receive a notice from the registrar. 6) Transfer credits from your original department may be partially or fully applied to the new department. Note: Transfer is competitive and depends on available slots."
    ))
    records.append(qa(
        "What is the tipping culture in Taiwan?",
        "Tipping is **not** customary in Taiwan. Unlike in the US or Canada, there is no expectation to tip at restaurants, taxis, hotels, or bars. Some high-end restaurants may include a 10% service charge in the bill — this is not a tip but a mandatory charge. Convenience stores, cafes, and food stalls do not expect tips. Rounding up the fare for a taxi driver is a gesture of goodwill but not required. For exceptional service at a hotel, a small tip (NT$100-200) to the bellhop or housekeeper is appreciated but never expected."
    ))
    records.append(qa(
        "How do I get from NTNU to Taichung or Kaohsiung on a budget?",
        "Budget travel from NTNU: **To Taichung**: Take the THSR (NT$765 standard, 50 min) or a bus from Taipei Bus Station (NT$250-350, 2 hours). Buses are cheaper but slower. **To Kaohsiung**: THSR (NT$1,490, 1h45min) or bus (NT$500-600, 4-5 hours). Book THSR early-bird tickets for up to 35% off. Bus companies: Ubus (統聯), Ho-Hsin (和欣), Kuo-Kuang (國光). All depart from Taipei Bus Station (next to Taipei Main Station, 10-min MRT from Guting). Book online or buy at the station."
    ))
    records.append(qa(
        "What should I do if I lose my NTNU student ID card?",
        "Lost student ID procedure: 1) Report the loss immediately through the NTNU student portal to disable card access and prevent misuse. 2) Visit the registrar's office (教務處) in person with your passport or ARC. 3) Fill out a re-issuance application. 4) Pay the replacement fee (NT$200-300). 5) A new card is issued within 1-2 hours (during business hours) or 1-2 days if the system is offline. 6) Reactivate your EasyCard balance by registering the new card at an MRT station machine. While waiting for the new card, you can get a temporary paper ID from the registrar for exam purposes."
    ))
    records.append(qa(
        "What is the academic calendar structure at NTNU?",
        "NTNU's academic year: **Fall semester** (第一學期) — mid-September to late January, with final exams in January. **Spring semester** (第二學期) — mid-February to late June, with final exams in June. **Summer break** — July to mid-September (about 10 weeks). **Winter break** — about 4 weeks between semesters (late January to mid-February). Classes run Monday to Friday, usually 8:10 AM to 6:20 PM (10 class periods of 50 minutes each). Evening classes run 6:30-9:30 PM. Most courses meet once a week for 2-3 periods."
    ))
    records.append(qa(
        "How does the NTNU cafeteria payment system work?",
        "NTNU cafeterias accept: 1) **EasyCard** — most convenient, just tap and go. 2) **Cash** — accepted at all counters (carry small bills and coins). 3) **Student ID card** — some cafeterias allow prepaid meal plans. 4) **Mobile payment** — Line Pay and JKOPay accepted at select vendors. The main student cafeteria at Heping Campus uses a tray system: pick your dishes (rice, main, sides), scan at the register, and pay. Average meal cost: NT$60-120. Some vendors offer meal deals (套餐) at NT$100-150 including a drink."
    ))
    records.append(qa(
        "Is there an NTNU mobile app for students?",
        "Yes, NTNU has a mobile app called 'NTNU Life' (師大生活) available for iOS and Android. Features include: 1) Class schedule and course info. 2) Academic calendar. 3) Campus map with building locations. 4) Library services (search, renew, reserve). 5) Cafeteria menus and hours. 6) Emergency contact numbers. 7) News and announcements. 8) Personal academic records (grades, attendance). The app is in Chinese but most functions are straightforward. Download from the App Store or Google Play by searching 'NTNU 生活'."
    ))
    records.append(qa(
        "How do I get a taxi in Taipei and what should I know?",
        "Getting a taxi in Taipei: 1) **Hailing** — wave on the street (empty cabs have a red '空車' sign). 2) **Apps** — Uber, FindTaxi, and Taiwan Taxi (台灣大車隊) app. Uber is more expensive but convenient. 3) **Phone** — call 55178 (Taiwan Taxi) or 0800-055-850 (English service available). 4) **MRT stations** — taxi stands at major stations. Fares: starting NT$100 for first 1.25km, then NT$5 per 200m after. Night surcharge: +NT$20 (11 PM-6 AM). Always carry cash; most taxis accept EasyCard but not credit cards. Drivers rarely speak English, so have your destination written in Chinese."
    ))

    return records
