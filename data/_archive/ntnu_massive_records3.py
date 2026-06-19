"""
Massive Taiwan-themed Q&A records (250+ pairs).

Covers six major sections: Politics & Government, Economy, Technology & Science,
Transportation, Food & Night Markets, and Nature & Environment.
"""

from typing import Dict, List
from ntnu_extended_records import qa


def get_massive_records3() -> List[Dict]:
    """Generate approximately 250+ Taiwan-themed Q&A records.

    Returns:
        List of ChatML-format record dicts.
    """
    records: List[Dict] = []

    # =====================================================================
    # Section 1: Taiwan Politics & Government (~50 records)
    # =====================================================================

    # -- Government Structure --
    records.append(qa(
        "台灣的政治體制是什麼？",
        "中華民國（台灣）是一個民主共和國，採行五權憲法體制，設有行政院、立法院、司法院、考試院及監察院五院。總統為國家元首，由人民直接選舉產生，任期四年，連選得連任一次。行政院為最高行政機關，院長由總統任命；立法院為最高立法機關，立法委員由人民直選產生；司法院負責解釋憲法與審判；考試院掌理公務人員考試與銓敘；監察院行使彈劾、糾舉與審計權。"
    ))
    records.append(qa(
        "What is the political system in Taiwan?",
        "Taiwan (Republic of China) operates under a democratic system based on the Five-Power Constitution, which separates governmental powers into five branches: the Executive Yuan, Legislative Yuan, Judicial Yuan, Examination Yuan, and Control Yuan. The President is the head of state, directly elected by the people for a four-year term renewable once. The Premier (President of the Executive Yuan) is appointed by the President. The Legislative Yuan is the supreme legislative body with members directly elected. This system was originally designed by Dr. Sun Yat-sen and is unique among modern democracies."
    ))
    records.append(qa(
        "中華民國總統的職權有哪些？",
        "中華民國總統為國家元首，對外代表國家。主要職權包括：統率全國陸海空軍（三軍統帥）、公布法律與發布命令、締結條約及宣戰媾和、宣布戒嚴、行使大赦特赦與減刑、任免文武官員、授予榮典。總統亦負責提名行政院長、司法院大法官、考試委員、監察委員及審計長，經立法院同意後任命。在立法院通過不信任案後，總統得解散立法院。"
    ))
    records.append(qa(
        "行政院在台灣政府中的角色是什麼？",
        "行政院是中華民國最高行政機關，相當於其他國家的內閣。院長（俗稱閣揆）為最高行政首長，由總統任命。行政院下設內政、外交、國防、財政、教育、法務、經濟、交通等八部，以及文化部、衛福部、勞動部、農業部、環境部、數位發展部等，加上國家發展委員會、大陸委員會等主要委員會。行政院會議為決策核心，決定重大施政方針。"
    ))
    records.append(qa(
        "立法院的立法委員如何產生？",
        "立法院為中華民國最高立法機關，現有113席立法委員，任期四年。其中73席為區域立法委員，由單一選區直選產生（一區一席）；6席為原住民立法委員（平地原住民3席、山地原住民3席）；34席為全國不分區及僑居國外國民立法委員，依政黨投票比例分配。立法委員候選人年滿23歲即可參選。"
    ))
    records.append(qa(
        "司法院大法官有什麼功能？",
        "司法院大法官共有15人，由總統提名，經立法院同意後任命，任期八年，不得連任。大法官的主要職權是解釋憲法與統一解釋法律及命令。大法官以會議方式合議審理憲法訴訟案件，包括法規範憲法審查、裁判憲法審查、政黨違憲解散案件等。大法官會議做出的解釋具有拘束全國各機關及人民的效力。"
    ))
    records.append(qa(
        "考試院和監察院分別掌理什麼事務？",
        "考試院為國家最高考試機關，掌理公務人員考試、銓敘、保障、撫卹、退休及人事行政等事項。設院長、副院長各一人及考試委員七至九人，由總統提名、立法院同意後任命。監察院為國家最高監察機關，行使彈劾、糾舉及審計權。設監察委員29人，由總統提名、立法院同意任命，院長由監察委員互選產生。審計長則負責審核政府決算。"
    ))
    records.append(qa(
        "Taiwan's government has five branches. What are they?",
        "Taiwan's Five-Power Constitution, based on Dr. Sun Yat-sen's political theory, separates power into five branches (Yuans): the Executive Yuan (administration), Legislative Yuan (legislation), Judicial Yuan (judiciary), Examination Yuan (civil service exams and personnel management), and Control Yuan (auditing and impeachment). This system adds two branches—Examination and Control—beyond the traditional three branches found in most Western democracies."
    ))
    records.append(qa(
        "中華民國憲法增修條文對政府體制有什麼影響？",
        "中華民國憲法歷經多次增修，主要變革包括：總統由國民大會間接選舉改為人民直接選舉（1996年首次實施）、凍結省級自治選舉（精省）、調整立法委員總額與選舉制度、司法院大法官改為總統提名立法院同意任命、考試院與監察院組織簡化。增修條文一方面強化總統權力，另一方面調整五院體制以適應民主化需求。"
    ))

    # -- Political Parties --
    records.append(qa(
        "台灣的主要政黨有哪些？",
        "台灣目前的主要政黨包括：民主進步黨（民進黨/DPP），2000年起多次執政，現任總統賴清德即為民進黨籍；中國國民黨（KMT），台灣歷史最悠久的政黨，曾長期執政；台灣民眾黨（TPP），2019年由柯文哲創立，為第三勢力主要代表；時代力量（NPP），2015年成立的進步政黨。此外還有台灣基進、綠黨、親民黨、新黨等小黨。"
    ))
    records.append(qa(
        "民主進步黨（DPP）的成立背景和主要政策主張？",
        "民主進步黨成立於1986年9月28日，是台灣第一個本土創建的政黨。創黨初期成員多為黨外運動人士，主張民主化與台灣獨立。民進黨為台灣主流政黨之一，政治立場偏向中間偏左，支持台灣主權獨立、社會福利、人權保障及能源轉型。民進黨於2000年至2008年首次執政（陳水扁），2016年至今再度執政（蔡英文、賴清德）。"
    ))
    records.append(qa(
        "中國國民黨（KMT）目前的立場和政策是什麼？",
        "中國國民黨由孫中山於1919年創立，1949年遷台後長期執政至2000年。國民黨主張一中憲法下的兩岸和平交流，支持九二共識，強調中華民國主權。經濟上偏向右派，支持自由貿易與企業發展。現任主席為朱立倫。近年國民黨在年輕世代與都會選區的支持度有所下降，但在地方選舉中仍具競爭力。"
    ))
    records.append(qa(
        "台灣民眾黨（TPP）在台灣政治中的角色是什麼？",
        "台灣民眾黨由柯文哲於2019年8月創立，定位為超越藍綠的第三勢力政黨。黨名致敬台灣先賢蔣渭水1927年創立的台灣民眾黨。TPP主張財政紀律、聯合政府、國家治理效率化。在2024年總統大選中，柯文哲獲得約26%選票，展現第三勢力的崛起。民眾黨在立法院擁有8席，扮演關鍵少數角色。"
    ))
    records.append(qa(
        "時代力量（NPP）是什麼樣的政黨？",
        "時代力量成立於2015年1月，由太陽花學運後的公民團體與進步學者共同創立。政治立場偏向左派進步主義，強調社會正義、人權保障、轉型正義與台灣主權。時代力量在2016年立法院選舉獲得5席，一度成為第三大黨，但後續因內部分歧與民眾黨崛起而影響力下降。"
    ))
    records.append(qa(
        "What are the main political parties in Taiwan?",
        "Taiwan's major political parties include the Democratic Progressive Party (DPP), which currently holds the presidency under Lai Ching-te; the Kuomintang (KMT), Taiwan's oldest party that governed for over five decades until 2000; and the Taiwan People's Party (TPP), founded in 2019 by Ko Wen-je as a third-force alternative. Smaller parties include the New Power Party (NPP), Taiwan Statebuilding Party, and the Green Party Taiwan. The party system has evolved from a KMT-dominated one-party state to a competitive multi-party democracy."
    ))

    # -- Key Political Figures --
    records.append(qa(
        "賴清德總統的政治經歷與政策方向？",
        "賴清德（William Lai），1959年出生，民進黨籍，2024年當選中華民國第16任總統。賴清德曾任國大代表、立法委員、台南市長（2010-2017），以治水與城市行銷聞名。2017年出任行政院長，2019年因民進黨初選爭議辭職。2020年當選副總統，並於2024年以40%得票率當選總統。賴清德被視為民進黨內新潮流系要角，政見強調台灣主權、經濟安全及社會正義。"
    ))
    records.append(qa(
        "蔡英文前總統的執政重點與貢獻？",
        "蔡英文（Tsai Ing-wen），1956年出生，為中華民國第14、15任總統（2016-2024），也是台灣第一位女性總統。蔡英文為法學博士，曾任陸委會主委與行政院副院長。任內重要政策包括：同性婚姻合法化（2019年）、推動能源轉型（2025非核家園）、促進半導體產業發展、推出國民年金改革、強化國防自主（國艦國造）。她在2020年以817萬票連任，創下台灣總統選舉史上最高得票紀錄。"
    ))
    records.append(qa(
        "馬英九前總統在兩岸關係上的主要政策是什麼？",
        "馬英九（Ma Ying-jeou），2008年至2016年擔任中華民國總統，國民黨籍。任內兩岸政策以九二共識為基礎，推動兩岸經貿與交流，包括簽署ECFA（兩岸經濟合作架構協議）、開放陸客來台觀光、兩岸直航（2008年啟動）。外交上採取活路外交，與中國達成外交休兵默契。但2014年太陽花學運後其兩岸政策面臨強烈民意挑戰。"
    ))
    records.append(qa(
        "柯文哲在台灣政治中的影響力如何？",
        "柯文哲（Ko Wen-je），1959年出生，前台大醫院創傷醫學部主任。2014年以無黨籍參選台北市長並當選，2018年連任。2019年創立台灣民眾黨並任黨主席。他以政治素人形象崛起，主張超越藍綠、務實治理。2024年代表民眾黨參選總統失利後，於2024年底因京華城案件遭羈押，引發政治爭議與支持者動員。"
    ))
    records.append(qa(
        "蔣萬安現任台北市長的背景？",
        "蔣萬安，1977年出生，國民黨籍，2022年當選台北市長（第七屆直轄市長）。他是蔣中正家族第四代，美國賓州大學法學博士，曾任執業律師。2016年首度參選立法委員即當選，2020年連任成功。在立委任內關注社福、勞工與教育議題。2022年擊敗民進黨陳時中、無黨籍黃珊珊當選台北市長。"
    ))
    records.append(qa(
        "陳水扁在台灣政治史上的地位？",
        "陳水扁，1950年出生，律師出身，因美麗島事件辯護律師而踏入政壇。1994年當選台北市長，2000年代表民進黨當選中華民國第10任總統，完成台灣首次政黨輪替。2004年以些微差距連任成功。任內推動台灣正名運動、公民投票法、國營事業民營化等政策。2008年卸任後因貪汙案入獄，2020年獲准保外就醫。他對台灣民主化與本土化的影響深遠且具爭議性。"
    ))
    records.append(qa(
        "李登輝對台灣民主化的貢獻是什麼？",
        "李登輝（1923-2020），台灣第一位本省籍總統，1988年蔣經國逝世後繼任，1996年當選首任民選總統。任內推動修憲、終止動員戡亂時期、廢除萬年國會、推動總統直選，被譽為台灣民主化之父。他在兩岸關係上提出特殊的國與國關係（兩國論）。晚年政治立場轉向本土化，支持台灣主權。李登輝對台灣從威權統治轉型為民主政治具有關鍵影響。"
    ))
    records.append(qa(
        "Who are the key political figures in Taiwan's recent history?",
        "Key political figures include: Tsai Ing-wen (蔡英文), Taiwan's first female president (2016-2024) who legalized same-sex marriage; Lai Ching-te (賴清德), the current president elected in 2024; Ma Ying-jeou (馬英九), president from 2008-2016 who improved cross-strait relations; Chen Shui-bian (陳水扁), the first DPP president (2000-2008); Lee Teng-hui (李登輝), the 'father of Taiwan's democratization'; Ko Wen-je (柯文哲), founder of the Taiwan People's Party; and Han Kuo-yu (韓國瑜), a polarizing KMT figure known for his populist style."
    ))

    # -- Electoral System --
    records.append(qa(
        "台灣的選舉制度是怎樣的？",
        "台灣選舉制度分為中央與地方層級。總統副總統由公民直接選舉，採相對多數制（一輪投票）。立法委員選舉採混合制：區域立委採單一選區相對多數決（73席），原住民立委採複數選區制（6席），不分區立委依政黨得票比例分配（34席，門檻5%）。地方首長（縣市長等）採相對多數制。縣市議員則採複數選區單記不可讓渡制。"
    ))
    records.append(qa(
        "台灣什麼時候首次舉辦總統直選？",
        "中華民國首次總統直選於1996年3月23日舉行，由李登輝（國民黨）以54%得票率當選。此前總統由國民大會間接選舉產生。1996年的總統直選是台灣民主化的重要里程碑，也是中華民國憲法增修後的重要制度變革。此後每四年舉行一次總統直選。"
    ))
    records.append(qa(
        "What is the voting age in Taiwan?",
        "The voting age in Taiwan is 20 years old for all elections, including presidential, legislative, and local elections. However, for referendums (citizen initiatives), the voting age was lowered to 18 in 2018. There have been ongoing debates about lowering the general voting age to 18, with constitutional amendments proposed but not yet fully ratified."
    ))
    records.append(qa(
        "台灣的公民投票制度是怎樣的？",
        "台灣的公民投票（公投）制度源自2003年通過的《公民投票法》。2021年公投修法後，公投與大選脫鉤、提案門檻調整。公投分為全國性與地方性兩類。2018年首次與地方選舉合併舉行，通過了反同婚公投、以核養綠公投等七案。2022年修憲公投將18歲公民權入憲。公投門檻為有效同意票超過不同意票，且同意票達投票權人總額四分之一以上。"
    ))

    # -- Local Government --
    records.append(qa(
        "台灣的地方政府層級有哪些？",
        "台灣的地方行政區劃分為三級：第一級為直轄市（六都）和縣市，第二級為鄉鎮市區，第三級為村里。六都包括：台北市、新北市、桃園市、台中市、台南市、高雄市。其餘縣市包括基隆市、新竹縣市、苗栗縣、彰化縣、南投縣、雲林縣、嘉義縣市、屏東縣、宜蘭縣、花蓮縣、台東縣、澎湖縣、金門縣、連江縣。直轄市長與縣市長皆由民選產生，任期四年。"
    ))
    records.append(qa(
        "六都之中哪個人口最多？",
        "新北市是台灣人口最多的直轄市，人口約400萬。其次為台中市（約280萬）、高雄市（約270萬）、桃園市（約230萬）、台北市（約250萬）、台南市（約180萬）。六都總人口約佔全台人口七成以上。"
    ))
    records.append(qa(
        "六都與其他縣市在權力上有何不同？",
        "直轄市（六都）在法律上享有較大的自治權限與較高的財政分配。直轄市政府的組織編制較大，局處設置較多，可自行管理的事務範圍也較廣。直轄市的社會福利、教育經費與基礎建設預算通常較縣市豐裕。2010年台灣進行縣市改制，部分縣市合併升格為直轄市，形成了目前的六都格局。"
    ))

    # -- Cross-Strait Relations --
    records.append(qa(
        "兩岸關係的現狀與主要爭議是什麼？",
        "兩岸關係目前處於官方對話中斷的狀態。主要爭議在於主權定位：中華民國政府主張台灣是主權獨立國家，中華人民共和國政府則主張台灣是中國的一部分。1992年的九二共識曾作為兩岸對話基礎，但近年雙方對此共識的解釋出現歧異。蔡英文政府任內兩岸官方交流停滯，但經貿往來持續。2024年賴清德就任後，兩岸關係仍面臨挑戰。"
    ))
    records.append(qa(
        "What is the One-China principle and how does it affect Taiwan?",
        "The One-China principle is Beijing's official position that there is only one China in the world, and Taiwan is part of China. This principle affects Taiwan's international relations, limiting its participation in organizations like the UN and WHO. Most countries maintain diplomatic relations with Beijing rather than Taipei, though many maintain unofficial ties through trade offices. Taiwan's government rejects the One-China principle as defined by Beijing, asserting its own sovereignty."
    ))
    records.append(qa(
        "九二共識的內容與爭議是什麼？",
        "九二共識是1992年兩岸在香港會談達成的默契，雙方各自以口頭方式表述「一個中國」原則。國民黨解讀為「一中各表」，即雙方都承認一個中國，但各自表述。民進黨則不承認九二共識的存在，認為其是國民黨單方建構的說法。2020年後，北京不再接受「一中各表」，強調「一個中國」就是中華人民共和國，九二共識的模糊空間大幅縮小。"
    ))
    records.append(qa(
        "台灣的邦交國有多少個？",
        "截至2025年，中華民國（台灣）的正式邦交國約有12個，主要位於中美洲、加勒比海地區、南太平洋及非洲。主要邦交國包括：巴拉圭、帛琉、馬紹爾群島、吐瓦魯、史瓦帝尼等。台灣與多數國家保持實質非官方關係，透過代表處或辦事處進行交流。"
    ))
    records.append(qa(
        "What is the Taiwan Relations Act?",
        "The Taiwan Relations Act (TRA), enacted by the U.S. Congress in 1979, provides the legal framework for unofficial relations between the U.S. and Taiwan after the U.S. switched diplomatic recognition to the PRC. The TRA authorizes the American Institute in Taiwan (AIT) to conduct relations and commits the U.S. to provide Taiwan with arms for self-defense. It remains a cornerstone of U.S.-Taiwan relations."
    ))
    records.append(qa(
        "台灣如何參與國際組織？",
        "由於中國的外交壓力，台灣無法以正式會員國身份參加聯合國及其附屬機構，但以觀察員或其他名義參與部分國際組織。台灣以中華台北名義參加奧運、亞運等體育賽事，以台灣、澎湖、金門及馬祖個別關稅領域（簡稱台灣）名義參加WTO世界貿易組織。台灣也以觀察員身份參與世界衛生大會（WHA）及國際民航組織（ICAO）部分活動。"
    ))

    # -- Military and Defense --
    records.append(qa(
        "台灣的國防政策與兵力概況？",
        "台灣實施義務役與志願役並行的兵役制度。2024年起義務役恢復為一年期。國軍總兵力約16萬人，分為陸軍、海軍、空軍、憲兵及海軍陸戰隊。國防政策以防衛固守、有效嚇阻為原則，強調不對稱作戰。近年重要國防政策包括：國艦國造（潛艦國造首艦2025年下水）、F-16V戰機升級、M1A2T戰車採購、增程型雄風反艦飛彈量產。"
    ))
    records.append(qa(
        "台灣的兵役制度2024年有什麼改革？",
        "2024年1月起，台灣恢復一年期義務役，適用於2005年1月1日以後出生的役男。義務役薪資大幅調升至約2萬元。訓練內容包括實彈射擊、城鎮戰訓練及新式武器操作。同時志願役官兵待遇亦獲提升，以吸引高素質人力。替代役仍保留，但條件更加嚴格。"
    ))
    records.append(qa(
        "What is Taiwan's defense strategy?",
        "Taiwan's defense strategy is based on asymmetric warfare (不對稱作戰), aiming to deter Chinese invasion through cost-imposing strategies rather than matching China quantitatively. Key elements include coastal defense with anti-ship missiles, air defense systems (Patriot, Tien-kung), and anti-landing operations. Taiwan also focuses on cyber defense, civil defense training, and upgrading domestic weapons systems."
    ))
    records.append(qa(
        "台灣的漢光演習是什麼？",
        "漢光演習（Han Kuang Exercise）是國軍年度最大的實兵操演，自1984年開始每年舉行。演習目的在驗證國軍聯合作戰能力與國土防衛計畫。2024年漢光40號演習取消以往的大規模火力展示模式，改採實戰化、無劇本演練，強化全民防衛與後備動員。演習涵蓋戰力保存、海上截擊、灘岸防衛與城鎮戰等科目。"
    ))

    # -- Human Rights and Social Movements --
    records.append(qa(
        "台灣的同性婚姻合法化過程是怎樣的？",
        "2019年5月17日，立法院通過《司法院釋字第七四八號解釋施行法》，使台灣成為亞洲第一個同性婚姻合法的國家。此法案源於2017年大法官釋字748號解釋，要求立法機關在兩年內完成立法。法案賦予同性伴侶結婚、財產繼承、醫療決定等權利。2023年更進一步開放跨國同性婚姻。"
    ))
    records.append(qa(
        "台灣的太陽花學運對社會有什麼影響？",
        "2014年3月18日爆發的太陽花學運是因立法院審查《海峽兩岸服務貿易協議》引發的大規模學生抗議。學生佔領立法院議場長達24天，最終迫使政府退回服貿協議。學運促進了年輕世代的政治參與，催生了時代力量等新政黨，也深化了公民對兩岸協議審查程序的監督意識。"
    ))
    records.append(qa(
        "台灣的女權運動發展狀況如何？",
        "台灣的女權運動自1980年代解嚴前後蓬勃發展。重要立法成就包括：2002年《兩性工作平等法》、2004年《性別平等教育法》。台灣女性勞動參與率約51%，高等教育女性比例已超越男性。2024年立法院女性立委比例約42%，高於全球平均。但同工同酬、職場玻璃天花板及家務分工不均等問題仍待改善。"
    ))
    records.append(qa(
        "台灣的原住民權利運動有哪些重要成果？",
        "台灣原住民族有16個官方承認的族群，約58萬人，佔總人口2.5%。2005年制定《原住民族基本法》，成立原住民族委員會。2016年蔡英文總統代表政府向原住民道歉。2017年《原住民族語言發展法》將原住民族語列為國家語言。重要運動包括傳統領域劃設、還我土地運動及蘭嶼達悟族反核廢料運動。"
    ))
    records.append(qa(
        "What major social movements have shaped modern Taiwan?",
        "Modern Taiwan has been shaped by several major social movements: the Sunflower Movement (2014) opposing the Cross-Strait Service Trade Agreement; the White Shirt Movement (2013) demanding military reform; the LGBTQ+ movement culminating in same-sex marriage legalization (2019); and various environmental movements protesting nuclear power and air pollution. These movements reflect Taiwan's vibrant civil society."
    ))
    records.append(qa(
        "台灣的勞工運動有哪些重要事件？",
        "台灣勞工運動歷史悠久，重要事件包括：1987年解嚴後工運蓬勃發展，1989年遠化工廠抗爭與1990年代自主工會運動。2011年華隆員工關廠抗爭、2014年國道收費員抗爭。近年焦點為勞動基準法修法爭議（一例一休、2017年勞基法修正案）、基本工資調整。台灣勞動者基本工資從2012年的NT$18,780調漲至2025年的NT$28,590。"
    ))

    # =====================================================================
    # Section 2: Taiwan Economy (~50 records)
    # =====================================================================

    # -- Economic History --
    records.append(qa(
        "台灣的經濟發展歷程是怎樣的？",
        "台灣經濟發展可分為幾個階段：1950年代進口替代工業化；1960-70年代出口導向工業化，加工出口區設立；1980年代產業轉型，科技產業崛起；1990年代高科技產業擴張，台積電等半導體公司成長；2000年後服務業佔GDP比重超過70%。人均GDP從1960年代的約200美元成長至2024年的約3.5萬美元。台灣被譽為亞洲四小龍之一。"
    ))
    records.append(qa(
        "台灣在1960-70年代的經濟起飛是如何實現的？",
        "1960年代台灣透過進口替代轉向出口導向政策，設立高雄、楠梓、台中加工出口區，吸引外資設廠生產紡織、電子產品出口。政府推動十大建設（1974-1979），包括中山高速公路、桃園國際機場、鐵路電氣化、高雄造船廠、石化業等基礎建設。年均GDP成長率超過9%，被譽為亞洲四小龍之一。"
    ))
    records.append(qa(
        "What is the 'Taiwan Miracle'?",
        "The 'Taiwan Miracle' refers to Taiwan's rapid industrialization and economic growth from the 1960s through the 1990s. Key factors included strong government planning (export processing zones, industrial parks), high savings and investment rates, land reform, universal education, and SME dynamism. Taiwan achieved average annual GDP growth of over 8% for three decades, earning its place among the Four Asian Tigers."
    ))
    records.append(qa(
        "台灣的加工出口區在經濟發展中扮演什麼角色？",
        "台灣於1966年設立全球第一個加工出口區——高雄加工出口區，隨後增設楠梓與台中加工區。這些區域提供完善基礎設施、簡化行政程序與稅務優惠，吸引外資設廠。加工區以勞力密集的電子與紡織業為主，鼎盛時期就業人數超過10萬人，出口額佔台灣總出口10%以上，為台灣經濟起飛提供重要動力。"
    ))
    records.append(qa(
        "十大建設對台灣經濟的影響是什麼？",
        "十大建設是1974年至1979年間由時任行政院長蔣經國推動的一系列重大基礎建設項目，包括：中山高速公路、桃園國際機場、鐵路電氣化、北迴鐵路、蘇澳港、台中港、高雄造船廠、大煉鋼廠（中鋼）、石油化學工業（中油）及核能發電廠。十大建設奠定了台灣現代化交通與工業基礎，促進了區域均衡發展與產業升級。"
    ))
    records.append(qa(
        "台灣如何在亞洲金融風暴中倖存？",
        "1997-1998年亞洲金融風暴重創泰國、韓國、印尼等國，但台灣受到的衝擊相對較小。關鍵因素包括：台灣外匯存底充足、外債比例低、銀行體系相對保守（放款審慎）、經濟基本面穩健（經常帳順差）。台灣央行當時穩定匯市，新台幣貶值幅度控制在20%以內，遠低於韓圜的50%以上。此經驗強化了台灣維持穩健總體經濟政策的共識。"
    ))

    # -- TSMC and Semiconductor --
    records.append(qa(
        "台積電（TSMC）在台灣經濟中扮演什麼角色？",
        "台積電是全球最大的晶圓代工廠，成立於1987年，2024年市值突破1兆美元。台積電佔台股加權指數權重約28%，營收約佔台灣GDP的8-10%。台積電在台灣設有竹科、中科、南科等生產基地，全球員工約7.5萬人。台積電的成功帶動了台灣半導體供應鏈的完整生態系。"
    ))
    records.append(qa(
        "台積電的創辦人是誰？公司是如何創立的？",
        "台積電由張忠謀於1987年在新竹科學園區創立。張忠謀出生於浙江寧波，曾在美國德州儀器擔任副總裁。在時任行政院長孫運璿與工研院院長張忠謀推動下，工研院衍生出台積電。台積電開創了純晶圓代工模式（Pure-Play Foundry），不設計晶片，專注為客戶生產，徹底改變了全球半導體產業。張忠謀被譽為半導體教父，2018年退休。"
    ))
    records.append(qa(
        "台積電的3奈米和5奈米製程有什麼競爭優勢？",
        "台積電在先進製程領域居全球領導地位。5奈米製程於2020年量產，客戶包括蘋果、NVIDIA、AMD等大廠。3奈米（N3）於2023年量產，是全球首個量產的3奈米技術。台積電的競爭優勢在於卓越的製程良率、年研發經費超過50億美元、緊密的客戶合作關係，以及持續推進至2奈米與1.4奈米的技術藍圖。"
    ))
    records.append(qa(
        "What is TSMC's significance to the global semiconductor industry?",
        "TSMC is the world's largest dedicated semiconductor foundry, producing over 90% of the world's most advanced chips (5nm and 3nm). Its clients include Apple, NVIDIA, AMD, Qualcomm, and MediaTek. TSMC enables the global AI boom, powering everything from iPhone processors to NVIDIA AI accelerators. The company is considered critical infrastructure for the global technology supply chain."
    ))
    records.append(qa(
        "台灣有哪些重要的半導體封測公司？",
        "台灣是全球封裝測試產業的龍頭，主要公司包括：日月光投控（ASE Technology Holding），全球最大封測廠；力成科技（PTI），專注記憶體封測；矽品精密（SPIL），已與日月光合組控股公司；京元電子，專精晶圓測試；頎邦科技，專注驅動IC封測。台灣封測產業全球市佔率超過50%。"
    ))
    records.append(qa(
        "台灣半導體產業的完整供應鏈是怎樣的？",
        "台灣半導體產業擁有全球最完整的供應鏈生態系。上游包括IC設計（聯發科、瑞昱、聯詠等）、矽智財（力旺、M31）；中游為晶圓製造（台積電、聯電、力積電、世界先進）與專業光罩（台灣光罩）；下游為封裝測試（日月光、力成、京元電子）。此外還有設備（帆宣、弘塑）、材料（長春石化、勝一化工）、EDA工具等支援產業，形成半導體聚落效應。"
    ))

    # -- Electronics and IT --
    records.append(qa(
        "鴻海（富士康）在台灣和全球的角色是什麼？",
        "鴻海精密工業（Hon Hai Precision Industry），以富士康（Foxconn）品牌聞名，由郭台銘於1974年創立。鴻海是全球最大的電子專業代工（EMS）廠商，營收約2,000億美元，員工超過80萬人。主要客戶包括Apple、Sony、Cisco等。鴻海在中國設有大量工廠，近年逐步將產能分散至印度、越南與墨西哥。"
    ))
    records.append(qa(
        "聯發科（MediaTek）在IC設計領域的地位如何？",
        "聯發科技成立於1997年，為全球第四大IC設計公司。聯發科擅長手機晶片組（Dimensity天璣系列）、智慧電視晶片、物聯網晶片及WiFi晶片。2024年聯發科手機晶片出貨量全球第一，尤其在5G中階市場佔主導地位。聯發科是台灣市值第二大的半導體公司，僅次於台積電。"
    ))
    records.append(qa(
        "廣達電腦在筆記型電腦代工的地位？",
        "廣達電腦（Quanta Computer）由林百里於1988年創立，是全球最大的筆記型電腦代工廠。廣達是蘋果MacBook、戴爾、聯想等品牌的主要代工夥伴。廣達亦積極布局雲端伺服器與AI伺服器市場，為NVIDIA HGX系統的重要合作夥伴。廣達總部位於桃園，製造據點包括中國、美國、德國等地。"
    ))
    records.append(qa(
        "What are Taiwan's major electronics companies besides TSMC?",
        "Taiwan's electronics ecosystem includes: Hon Hai (Foxconn), world's largest electronics manufacturer; MediaTek, top-4 global chip designer; Quanta Computer, largest laptop ODM; Wistron, Compal, and Inventec (major PC/notebook ODMs); Pegatron (Apple supplier); ASE Technology (world's largest semiconductor packaging company); AUO and Innolux (display panels); Delta Electronics (power supplies)."
    ))
    records.append(qa(
        "台灣的自行車產業有什麼特色？",
        "台灣是全球高階自行車製造重鎮，產值約佔全球7-8成。巨大機械（Giant）是全球最大自行車製造商之一，捷安特品牌聞名全球。美利達（Merida）亦為國際知名品牌。台灣自行車產業擁有完整供應鏈，從碳纖維車架、變速系統到輪組皆可自製。近年電動輔助自行車（E-bike）成為成長最快的新業務。"
    ))
    records.append(qa(
        "華碩電腦的發展歷程與產品線？",
        "華碩電腦（ASUS）成立於1989年，由四位曾在宏碁工作的工程師創立。華碩從主機板起家，逐步發展為全球前五大個人電腦品牌。產品線包括：ZenBook與ROG系列筆電、ZenFone智慧型手機、ROG電競周邊、主機板與顯示卡。華碩的ROG（Republic of Gamers）電競品牌為全球電競硬體領導品牌之一。"
    ))
    records.append(qa(
        "宏碁在台灣科技產業的歷史地位？",
        "宏碁（Acer）由施振榮等人於1976年創立，是台灣最早的電腦公司之一，也是最早推動國際品牌（Acer、Gateway、Packard Bell）的台灣科技公司。宏碁在1990年代曾是全球前五大個人電腦品牌，對台灣科技產業的國際化啟蒙影響深遠。施振榮提出的微笑曲線理論（研發與品牌價值最高、製造中間）影響了整個台灣產業政策。"
    ))

    # -- Traditional Industries --
    records.append(qa(
        "台灣傳統製造業有哪些重要領域？",
        "台灣傳統製造業包括：石化業（台塑集團為代表）、鋼鐵業（中鋼）、紡織業（機能性布料全球知名，如遠東新世紀、儒鴻）、工具機業（全球出口前10大）、螺絲螺帽業（全球市佔第二）、製鞋業（寶成、豐泰為Nike等品牌代工）。這些產業近年積極導入自動化與智慧製造轉型。"
    ))
    records.append(qa(
        "台塑集團對台灣經濟的影響力？",
        "台塑集團由王永慶與王永在兄弟創立，為台灣最大石化集團。旗下主要公司包括台塑、南亞、台化、台塑石化（四寶）。台塑石化是台灣最大民營煉油廠，擁有日煉量54萬桶的六輕煉油廠。集團營收約佔台灣GDP的5%，雇用超過10萬名員工。從塑膠原料起家，垂直整合至石化上游、紡織、電子材料等領域。"
    ))
    records.append(qa(
        "What is the status of Taiwan's textile industry?",
        "Taiwan's textile industry has evolved from low-cost mass production to high-value functional textiles. Companies like Far Eastern New Century, Eclat Textile, and Roo Hsing produce performance fabrics for global brands like Nike, Adidas, and Lululemon. Taiwan leads in recycled polyester from PET bottles, waterproof/breathable fabrics, and eco-friendly dyeing technologies."
    ))
    records.append(qa(
        "中鋼在台灣工業發展中的角色？",
        "中國鋼鐵公司（中鋼）成立於1971年，為十大建設之一，是台灣唯一的一貫作業鋼鐵廠。中鋼年粗鋼產能約1,000萬噸，在台灣鋼鐵市場佔有率約50%。中鋼不僅提供建築、汽車、機械等產業所需的鋼材，也透過技術研發與人才培訓帶動台灣鋼鐵下游產業發展。中鋼也是台灣極少數經常獲利的國營事業之一。"
    ))

    # -- Service Sector --
    records.append(qa(
        "台灣的服務業現狀如何？",
        "服務業佔台灣GDP約70%，就業人口佔比約60%。主要服務業包括：金融保險業（約6.5%）、批發零售業（約15%）、資訊服務業、觀光餐飲業、醫療保健業、物流運輸業等。便利商店密度全球第二（僅次於韓國），全台超過1.2萬家。台灣服務業以中小企業為主，面臨數位轉型與人力短缺的挑戰。"
    ))
    records.append(qa(
        "台灣的便利商店文化有什麼特色？",
        "台灣便利商店密度極高，7-Eleven超過6,800家、全家約4,200家。提供多元服務：代收帳單、快遞收件、影印傳真、售票、ATM、咖啡、便當、關東煮等。7-Eleven的City Café年銷量超過4億杯。超商也成為包裹寄送與電商取貨的最重要渠道，創造獨特的消費生態。"
    ))
    records.append(qa(
        "台灣的醫療服務業有什麼國際競爭力？",
        "台灣醫療以高品質、低價格聞名。全民健康保險（健保）覆蓋率超過99%，醫療支出僅佔GDP約6.5%。台灣在肝臟移植、顯微手術、心血管治療及達文西手術方面具有國際水準。醫療旅遊興起，每年約30萬外國人來台就醫。長庚醫院、台大醫院、台北榮總為最具規模的醫學中心。"
    ))

    # -- Financial System --
    records.append(qa(
        "台灣中央銀行的職能是什麼？",
        "中華民國中央銀行（央行）為最高金融主管機關，職能包括：制定貨幣政策、發行貨幣、管理外匯存底（2024年約5,700億美元，全球第五）、監管支付系統、擔任政府銀行。現任總裁為楊金龍。央行運用利率政策與選擇性信用管制調控房市。台灣長期維持低通膨（約1-2%）與穩定匯率。"
    ))
    records.append(qa(
        "台灣有哪些主要的銀行？",
        "台灣銀行體系分為公股與民營銀行。主要公股銀行：台灣銀行、土地銀行、合作金庫、第一銀行、華南銀行、彰化銀行、兆豐銀行。民營銀行：國泰世華銀行、中國信託銀行、台北富邦銀行、玉山銀行、台新銀行。純網銀（LINE Bank、將來銀行）於2022年正式營運。"
    ))
    records.append(qa(
        "What is Taiwan's foreign exchange reserve position?",
        "Taiwan's foreign exchange reserves reached approximately $570 billion as of 2024, ranking fifth in the world after China, Japan, Switzerland, and India. Reserves are managed by the Central Bank and are held primarily in U.S. Treasury bonds and gold. Taiwan's large reserves reflect its persistent trade surplus, particularly from semiconductor exports."
    ))
    records.append(qa(
        "台灣的證交所與資本市場概況？",
        "台灣證券交易所（TWSE）成立於1961年，加權股價指數為主要指標。台股總市值約50兆新台幣（約1.6兆美元），排名全球第15左右。上市公司超過1,000家。台積電佔台股權重約28%，為最大成分股。2023年台灣推出創新板（TIB），鼓勵新創公司上市。台股以電子科技類股為主，佔比超過60%。"
    ))

    # -- Trade Relationships --
    records.append(qa(
        "台灣的主要貿易夥伴有哪些？",
        "台灣主要貿易夥伴：中國大陸（含香港）佔出口約35-40%，進口約20%；美國為第二大出口市場（約15%），因AI與半導體需求持續成長；日本為主要進口來源國（設備與零組件）；東協國家份額持續上升。韓國、歐洲、中東（能源）亦為重要貿易對象。台灣2024年總貿易額約8,000億美元。"
    ))
    records.append(qa(
        "台灣與中國的經濟依存關係如何？",
        "兩岸經貿密切但高度不對稱。台灣對中國（含香港）出口佔總出口約35-40%，其中半導體與電子零組件為大宗。台灣對中國投資累計超過2,000億美元。但因供應鏈移轉，台灣對中國出口依賴度已從2019年的約40%降至2024年的約35%。"
    ))
    records.append(qa(
        "台灣與美國的經貿關係近年有什麼發展？",
        "台美經貿關係近年持續深化。2023年簽署《台美21世紀貿易倡議》首批協定。美國是台灣第二大貿易夥伴，也是半導體設備與軍購主要來源。2024年台灣對美出口大幅成長，受AI伺服器需求帶動。台積電在亞利桑那州投資超過650億美元建設先進晶圓廠。"
    ))
    records.append(qa(
        "What is Taiwan's role in global supply chains?",
        "Taiwan is a critical node in global supply chains, particularly in semiconductors and electronics. Taiwan produces over 60% of global semiconductors, 90% of advanced chips (sub-7nm), and 80-90% of server manufacturing. Taiwanese companies are essential suppliers to Apple, NVIDIA, AMD, Intel, Dell, HP, and Cisco. This concentration is both an economic strength and a geopolitical vulnerability."
    ))

    # -- Startup Ecosystem --
    records.append(qa(
        "台灣的創業生態系統發展如何？",
        "台灣創業生態在近十年快速成長。台北被評為全球新創生態前40名。主要創業聚落包括台北忠孝東路、松山文創園區、新竹工研院周邊及台中軟體園區。國發基金每年投資約10億美元支持新創。台灣在AI、物聯網、生技醫療、FinTech及綠色科技領域湧現許多新創公司。"
    ))
    records.append(qa(
        "台灣有哪些知名的獨角獸公司？",
        "台灣獨角獸公司（估值超10億美元）包括：Appier（AI行銷科技，東京上市）、Gogoro（電動機車與電池交換系統，美國上市）、Kdan Mobile（軟體服務）、Pinkoi（設計商品電商平台）、ShopBack（現金回饋平台，有台灣團隊）。此外Grab與Lalamove等國際獨角獸也有台灣創辦人。"
    ))
    records.append(qa(
        "政府對新創公司的支援政策有哪些？",
        "政府透過國發基金、經濟部等機構支援新創。措施包括：創業天使投資方案、Startup Terrace林口新創園區、亞洲矽谷計畫、稅務優惠。科技部FITI創新創業激勵計畫每年選拔優秀團隊。各縣市設立育成中心與共同工作空間，提供創業輔導與資源對接。"
    ))

    # -- Economic Challenges --
    records.append(qa(
        "台灣面臨哪些主要的經濟挑戰？",
        "台灣經濟面臨的主要挑戰：人口老化與少子化（總生育率約0.87，全球倒數）、能源轉型（非核家園與供電穩定）、產業結構過度集中半導體、城鄉發展不均、青年低薪與房價高漲、兩岸關係不確定性、全球供應鏈重組壓力。"
    ))
    records.append(qa(
        "台灣的少子化問題對經濟有什麼影響？",
        "台灣總生育率僅約0.87，全球最低之一。少子化導致勞動力減少，預計2030年後勞動年齡人口每年減少約15萬人。學校面臨招生不足，農業與製造業缺工嚴重。政府推出擴大育兒補貼、提高育嬰留職停薪津貼、增加公立托育名額等政策但效果有限。"
    ))
    records.append(qa(
        "What are Taiwan's main economic challenges?",
        "Taiwan's key economic challenges include: the world's lowest birth rate (0.87), causing rapid population aging; over-reliance on semiconductors (TSMC alone accounts for ~8% of GDP); energy transition difficulties; high housing prices; stagnant youth wages; urban-rural income gaps; and geopolitical risks from cross-strait tensions."
    ))
    records.append(qa(
        "台灣的房價問題有多嚴重？",
        "台灣房價所得比（房價中位數/家庭年所得中位數）約9.8倍，台北市更高達16倍以上，為全球最難負擔的城市之一。政府推出多項打房措施：選擇性信用管制（央行限貸）、房地合一稅2.0（持有五年內交易課重稅）、實價登錄2.0（揭露完整門牌）、預售屋禁止轉售。但房價上漲趨勢仍未被完全抑制。"
    ))

    # -- Tourism --
    records.append(qa(
        "台灣的觀光產業有什麼特色？",
        "台灣觀光以美食、自然景觀、人文歷史與友善安全環境為特色。2019年疫情前國際觀光客達1,186萬人次，主要來源為日本、韓國、東南亞、港澳與歐美。著名景點包括台北101、故宮博物院、太魯閣、日月潭、阿里山、墾丁、九份老街。民宿文化發達，美食旅遊（夜市、小籠包、牛肉麵）為重要吸引力。"
    ))
    records.append(qa(
        "What makes Taiwan a popular tourist destination?",
        "Taiwan attracts tourists with its natural beauty (Taroko Gorge, Sun Moon Lake, Alishan, Kenting), vibrant night markets (Shilin, Raohe, Fengchia), world-class food (beef noodle soup, xiaolongbao, bubble tea), rich history, hot springs, and safety. The High Speed Rail makes island-wide exploration easy. Taiwan is considered one of the safest travel destinations in Asia."
    ))
    records.append(qa(
        "台灣的民宿文化有什麼特色？",
        "台灣民宿產業蓬勃發展，全台登記民宿超過1萬家。民宿文化強調在地特色與人情味，從宜蘭的田園民宿、花蓮的海景民宿、南投的山居民宿到墾丁的度假民宿，各有主題風格。政府推動好客民宿與星級民宿評鑑制度，保障住宿品質。疫情後國內旅遊盛行，民宿市場更為熱絡。"
    ))

    # =====================================================================
    # Section 3: Taiwan Technology & Science (~40 records)
    # =====================================================================

    # -- Hsinchu Science Park --
    records.append(qa(
        "新竹科學園區在台灣科技發展中的角色是什麼？",
        "新竹科學園區（竹科）成立於1980年，是台灣第一個科學園區，被稱為台灣矽谷。竹科佔地約1,400公頃，進駐超過600家高科技公司，從業人員超過17萬人。園區以半導體產業為核心，包括晶圓代工（台積電、聯電）、IC設計（聯發科、瑞昱）、光電（友達）等領域，年營業額超過1.5兆新台幣。"
    ))
    records.append(qa(
        "台灣的三大科學園區各有哪些特色？",
        "新竹科學園區（竹科）成立最早（1980年），以半導體與光電產業為主。中部科學園區（中科）成立於2003年，以光電、精密機械與半導體為核心，台積電先進製程進駐後產值大增。南部科學園區（南科）成立於1996年，近年台積電3奈米廠進駐後成為全球最先進半導體製造重鎮。三大園區總產值約3.5兆新台幣。"
    ))
    records.append(qa(
        "What is the Hsinchu Science Park?",
        "The Hsinchu Science Park (HSP), established in 1980, is Taiwan's first science park and the birthplace of its semiconductor industry. Located 70km southwest of Taipei, the park houses over 600 high-tech companies including TSMC, UMC, and MediaTek. It employs over 170,000 people with annual revenues exceeding NT$1.5 trillion. The park inspired similar parks in Central and Southern Taiwan."
    ))

    # -- Manufacturing Process --
    records.append(qa(
        "半導體製造的主要流程是什麼？",
        "半導體製造分為前端與後端。前端製程：設計（IC Design）、光罩製作（Mask Making）、晶圓製程（Wafer Fabrication），包括氧化、沉積、微影、蝕刻、摻雜等步驟重複數十次。後端製程：晶圓測試、切割、封裝與最終測試。先進製程（如3奈米）需超過1,000個製程步驟，生產週期約3-4個月。"
    ))
    records.append(qa(
        "台灣在先進半導體封裝技術方面有什麼進展？",
        "先進封裝已成半導體性能提升的關鍵。台積電的3D Fabric平台整合CoWoS與InFO等技術，用於NVIDIA AI晶片與Apple處理器封裝。日月光開發FoMCP、SiP等先進封裝技術。台灣在先進封裝領域全球市佔率約60%。"
    ))
    records.append(qa(
        "What is CoWoS packaging technology?",
        "CoWoS (Chip-on-Wafer-on-Substrate) is TSMC's advanced 2.5D packaging technology integrating multiple chips on a silicon interposer. It has become crucial for AI accelerators — NVIDIA's H100, B200, and future AI GPUs all use CoWoS. Demand from AI has made CoWoS capacity a bottleneck in global AI chip supply. The technology enables high-bandwidth chip-to-chip communication."
    ))

    # -- Notable Tech Companies --
    records.append(qa(
        "聯華電子（UMC）與台積電的差別在哪裡？",
        "聯華電子（UMC）成立於1980年，曾是台灣第一家半導體公司。與台積電專注先進製程不同，UMC在2018年宣布放棄7奈米以下先進製程研發，轉而專注成熟製程（28奈米以上）。UMC在特殊製程（高壓、嵌入式記憶體、MEMS等）具有優勢，並與美國、日本合作夥伴建立策略聯盟。"
    ))
    records.append(qa(
        "台灣有哪些知名的IC設計公司？",
        "除了聯發科（MediaTek）外，台灣還有許多優秀IC設計公司：瑞昱半導體（Realtek，網通與音頻晶片全球領先）、聯詠科技（Novatek，驅動IC與影像處理）、群聯電子（Phison，NAND快閃記憶體控制器全球龍頭）、新唐科技（Nuvoton，微控制器）、義隆電子（Elan，觸控晶片）、原相科技（PixArt，光學感測器）。"
    ))
    records.append(qa(
        "緯創與和碩在電子代工產業的角色？",
        "緯創資通（Wistron）與和碩聯合科技（Pegatron）均為台灣主要的電子專業代工（EMS）公司。兩家皆為蘋果供應鏈重要成員。緯創是全球前十大ICT代工廠之一。和碩原是華碩製造部門，2008年分拆獨立，現為蘋果iPhone與MacBook主要組裝合作夥伴之一。"
    ))

    # -- Science Education --
    records.append(qa(
        "台灣的科學教育在國際評比中表現如何？",
        "台灣學生在國際評比中表現優異。PISA 2022結果：數學排名第3、科學第4、閱讀第5。TIMSS中台灣學生在數學與科學領域均名列前茅。台灣設有科學班與科學資優班，每年舉辦國際科學展覽會，培養年輕科學人才。"
    ))
    records.append(qa(
        "台灣的國際科學奧林匹亞競賽表現如何？",
        "台灣在國際科學奧林匹亞競賽表現極為出色，至2024年累計超過700面獎牌（含約200面金牌）。地球科學奧林匹亞經常獲得世界第一，物理奧林匹亞長期團體前三，化學、生物與資訊奧林匹亞亦名列世界頂尖。反映台灣科學教育的扎實基礎。"
    ))

    # -- Research Institutes --
    records.append(qa(
        "中央研究院在台灣學術界的地位如何？",
        "中央研究院（Academia Sinica）成立於1928年，為中華民國最高學術研究機構，直屬於總統府。設有數理科學、生命科學、人文及社會科學三個學組，共約24個研究所與8個研究中心。中研院擁有超過20位諾貝爾獎得主院士。現任院長為廖俊智。中研院在基因體學、天文物理、台灣歷史研究等領域居國際領先地位。"
    ))
    records.append(qa(
        "工業技術研究院（ITRI）在台灣科技發展中的貢獻？",
        "工業技術研究院（ITRI/工研院）成立於1973年，是非營利應用研究機構，被譽為台灣產業升級引擎。工研院成功衍生出台積電、聯電等世界級公司。工研院在電子、機械、材料、能源、生醫等領域進行前瞻研發。2023年與超過1萬家廠商合作，每年產出超過1,500件專利。"
    ))
    records.append(qa(
        "What role does ITRI play in Taiwan's technology development?",
        "ITRI has been a cornerstone of Taiwan's technology development since 1973. It played a pivotal role by transferring IC manufacturing technology from RCA (USA) and later spinning off TSMC and UMC. ITRI continues driving innovation in AI, IoT, EV components, and green energy. It operates the Innovation Campus in Hsinchu, incubating hundreds of startups."
    ))
    records.append(qa(
        "國家實驗研究院的功能是什麼？",
        "國家實驗研究院（NARLabs）成立於2003年，隸屬國科會。旗下包括：國家晶片系統設計中心（CIC）、國家高性能計算中心（NCHC，擁有台灣杉超級電腦）、國家地震工程研究中心（NCREE，亞洲最大地震模擬振動台）、國家太空中心（TASA，管理福爾摩沙衛星計畫）、海洋科技研究中心等。"
    ))

    # -- Space Program --
    records.append(qa(
        "台灣的太空計畫有哪些成就？",
        "國家太空中心（TASA）成立於1991年，已成功發射多枚衛星。福衛一號1999年發射，為台灣首枚自主研發衛星。福衛二號提供高解析度遙測影像。福衛三號為全球首個氣象衛星星系。福衛五號為首枚自主光學遙測衛星。福衛七號提供精準氣象預報。2024年首枚自製通訊衛星獵風者號成功升空。"
    ))
    records.append(qa(
        "What is Taiwan's FORMOSAT satellite program?",
        "Taiwan's FORMOSAT program, managed by TASA, has launched multiple satellites since 1999. FORMOSAT-3 (2006) was a constellation of six microsatellites providing global atmospheric data, revolutionizing weather forecasting. FORMOSAT-5 (2017) was Taiwan's first domestically-built optical remote sensing satellite. FORMOSAT-7 (2019) replaced FORMOSAT-3 with an advanced constellation co-developed with NOAA."
    ))
    records.append(qa(
        "台灣的火箭發射能力如何？",
        "台灣目前尚未具備中大型火箭發射能力，但TASA正在發展小型發射載具。2023年核准屏東縣牡丹鄉旭海村為台灣首個民間火箭發射場。晉陞太空科技（TiSPACE）等民間公司研發商用火箭。台灣太空政策以衛星技術為核心，透過國際合作進行衛星發射。"
    ))

    # -- Medical Technology --
    records.append(qa(
        "台灣的醫療科技有哪些成就？",
        "台灣在醫療科技領域有多項成就：全球第一台MRgFUS治療系統由台灣團隊參與開發；長庚醫院完成亞洲首例心肺同時移植；iPS細胞研究居國際前列。防疫科技方面，2020年快速研發口罩地圖、COVID-19檢測試劑。台灣健保資料庫是全球最大醫療資料庫之一，吸引國際學術合作。"
    ))
    records.append(qa(
        "台灣的智慧醫療發展現狀？",
        "台灣智慧醫療結合ICT優勢與醫療專業快速發展。主要應用：AI輔助診斷（台大、北榮導入AI判讀醫療影像）、遠距醫療（偏鄉服務）、電子病歷互通（健保雲端）、達文西手術系統（亞洲第二高使用密度）。多家科技公司投入智慧醫療設備研發，2024年產值超過1,000億新台幣。"
    ))
    records.append(qa(
        "台灣有哪些重要的生技醫藥公司？",
        "生技醫藥公司包括：藥華醫藥（P1101治療真性紅血球增多症，獲美歐藥證）、合一（ON101糖尿病傷口癒合藥物）、中天/泉盛、太景（抗生素新藥）、高端疫苗（COVID-19疫苗）。台灣生技產業年產值約1,000億新台幣，在細胞治療、新藥開發與醫療器材領域持續成長。"
    ))

    # -- Green Energy --
    records.append(qa(
        "台灣的再生能源發展目標與現狀？",
        "台灣再生能源發展目標為2025年占比20%（太陽光電20GW、離岸風電5.6GW）。至2024年，再生能源占比約9-10%。太陽光電已建置約12GW，離岸風電約2GW。台灣海峽擁有全球最佳風場條件，吸引沃旭能源、北陸能源等國際開發商。地熱與小水力發電也在推廣中。"
    ))
    records.append(qa(
        "台灣在離岸風電方面有什麼發展？",
        "台灣擁有全球最優良離岸風力資源（台灣海峽平均風速每秒12公尺以上）。政府推動區塊開發，目標2025年5.6GW、2035年20GW。2023年台灣離岸風電裝置容量達2.1GW，居亞太第二。主要風場包括海洋風電（Formosa 1）、大彰化風場（沃旭能源）等。台灣也建立本土供應鏈（水下基礎、風機塔架、海事工程）。"
    ))
    records.append(qa(
        "What is Taiwan's current energy mix?",
        "Taiwan's electricity generation mix as of 2024: natural gas ~45%, coal ~35%, nuclear ~6% (declining toward 2025 nuclear-free goal), renewables ~9% (solar ~5%, wind ~2%, hydro ~2%). Taiwan relies on imported energy for about 97% of supply. The 2025 targets include 20% renewable energy and 50% natural gas, but renewable buildout has been slower than planned."
    ))

    # -- Digital Transformation & AI --
    records.append(qa(
        "台灣的AI產業發展現狀如何？",
        "台灣AI產業以AI晶片製造與AI應用為核心。台積電為全球AI晶片主要生產者。政府推出AI行動計畫2.0，投入超過300億新台幣。重點領域包括AI醫療、AI製造、自駕車、智慧城市。台灣AI實驗室由杜奕瑾創辦，專注本土AI技術。微軟、Google、NVIDIA在台設立AI研發中心。"
    ))
    records.append(qa(
        "台灣的數位轉型政策有哪些？",
        "政府推動DIGI+方案與智慧台灣方案。主要措施：5G網路建設（2024年覆蓋率達95%以上）、智慧城鄉應用（智慧交通、智慧醫療、防災物聯網）、數位人才培育。2022年成立數位發展部（moda），統籌資訊安全、通訊傳播與數位產業發展。台灣政府數位服務在國際評比中名列前茅。"
    ))

    # =====================================================================
    # Section 4: Taiwan Transportation (~30 records)
    # =====================================================================

    # -- THSR --
    records.append(qa(
        "台灣高鐵的路線與營運狀況？",
        "台灣高速鐵路（THSR）全長350公里，2007年通車，連接台北與左營（高雄）。設有台北、板橋、桃園、新竹、苗栗、台中、彰化、雲林、嘉義、台南、左營共11站。營運最高時速300公里，台北至高雄最快約90分鐘。日運量約20萬人次，準點率超過99%。採用日本新幹線700T系列車技術。"
    ))
    records.append(qa(
        "台灣高鐵採用的技術與日本新幹線的關係？",
        "台灣高鐵核心技術源自日本新幹線系統，採用700T型列車，由川崎重工與日立製造。700T以日本新幹線700系列車為基礎，針對台灣亞熱帶氣候與地形改良（強化空調、抗腐蝕能力）。台灣高鐵營運管理導入新幹線的調度控制系統（CTC）與車輛檢修體系。"
    ))
    records.append(qa(
        "How fast does the Taiwan High Speed Rail go?",
        "The Taiwan High Speed Rail (THSR) operates at a maximum speed of 300 km/h (186 mph), covering the 350 km from Taipei to Kaohsiung in as little as 90 minutes. Based on Japan's Shinkansen 700 series, THSR maintains an on-time rate exceeding 99% and carries approximately 200,000 passengers daily across its 11 stations."
    ))
    records.append(qa(
        "台灣高鐵的票價結構與優惠方案有哪些？",
        "台北至左營標準車廂全票約NT$1,490，商務車廂約NT$2,120。優惠：早鳥票（最低65折）、大學生優惠（5折或75折）、定期票與回數票、團體票、敬老/身心障礙/孩童票（5折）。台灣高鐵推動常客方案與T-EX App行動支付，提供便利購票體驗。"
    ))

    # -- Taipei MRT --
    records.append(qa(
        "台北捷運的路網概況與運量？",
        "台北捷運1996年木柵線通車，現有五條主線與兩條支線，總長約152公里，131個車站。路線：文湖線（棕線）、淡水信義線（紅線）、松山新店線（綠線）、中和新蘆線（橘線）、板南線（藍線），及新北投與小碧潭支線。日運量約200萬人次。以整潔、準時、文明禮儀（禁止飲食、排隊文化）著稱。"
    ))
    records.append(qa(
        "台北捷運的票價與票證系統是什麼？",
        "台北捷運票價依里程計算，NT$20起跳，最高NT$65。主要使用悠遊卡感應支付，也可用手機支付（Apple Pay、Google Pay、LINE Pay等）及信用卡直接感應。2020年起改為QR Code單程票，全面啟用閘門直接刷卡進出。與YouBike、公車、台鐵等整合轉乘優惠。"
    ))
    records.append(qa(
        "What is the Taipei Metro like?",
        "The Taipei Metro is widely regarded as one of the world's best subway systems for its cleanliness, punctuality, and efficiency. With 5 main lines and 131 stations spanning 152 km, it carries about 2 million passengers daily. Features include strict no eating/drinking policies, clear multi-language announcements, platform screen doors, and excellent accessibility. It uses the EasyCard contactless system."
    ))
    records.append(qa(
        "高雄捷運的路線與運量概況？",
        "高雄捷運營運紅線（南岡山至小港，約28公里）與橘線（西子灣至大寮，約14公里），總長約42公里，38個車站。日運量約18萬人次。特色車站包括美麗島站（全球最美地鐵站之一，光之穹頂公共藝術）、中央公園站。高捷結合環狀輕軌形成更完整運輸網絡。"
    ))
    records.append(qa(
        "台灣還有哪些城市有捷運或輕軌系統？",
        "除台北與高雄外：桃園機場捷運（2017年通車，連結台北、機場至中壢）、桃園捷運綠線（興建中）、台中捷運綠線（2021年通車）、新北安坑輕軌與淡海輕軌（2018-2023年通車）、高雄環狀輕軌（2024年全線通車）。全台軌道運輸網絡持續擴張中。"
    ))

    # -- Taiwan Railway --
    records.append(qa(
        "台鐵的路線網絡與營運概況？",
        "台灣鐵路管理局（台鐵）成立於1887年，環島鐵路網總長約1,100公里，車站約240個。主要路線：縱貫線（基隆至屏東）、宜蘭線、北迴線、花東線、南迴線，構成環島鐵路。車種包括自強號（最快）、莒光號、復興號、區間車。日運量約65萬人次。2024年公司化轉型為國營公司。"
    ))
    records.append(qa(
        "台鐵的EMU3000型城際列車有什麼特色？",
        "EMU3000型由日立製作所與台灣車輛公司合作生產，共購置600輛。特色：最高速度130公里/小時、座椅寬敞舒適、全車配充電插座及WiFi、車廂設計融入台灣元素。主要服務西部幹線長途旅客，逐步取代老舊推拉式自強號。"
    ))
    records.append(qa(
        "What is the difference between Taoyuan Airport MRT and Taiwan Railway?",
        "The Airport MRT is a dedicated express line (2017) connecting Taipei Main Station, Taoyuan Airport, and Zhongli. It offers Express (35 min Taipei-Airport) and Commuter services. Taiwan Railway (TRA) is the conventional rail system running around the entire island, serving all major cities with cheaper fares and more frequent stops."
    ))
    records.append(qa(
        "台灣的東部鐵路改善計畫有哪些重要工程？",
        "東部鐵路改善計畫包含：北迴線電氣化（2014年完工）、花東線電氣化（2014年完工）、南迴鐵路電氣化（2020年完工，完成環島電氣化最後一哩）、花東線雙軌化（進行中）。台北至台東自強號行車時間從7小時縮短至約4小時。"
    ))

    # -- Highways --
    records.append(qa(
        "台灣的國道高速公路網絡概況？",
        "國道高速公路總長約1,100公里。主要路線：國道一號（中山高，基隆至高雄，373公里，1978年全線通車）、國道三號（北二高/福爾摩沙高速公路，432公里）、國道五號（北宜高含雪山隧道12.9公里，為東南亞最長公路隧道）。全面實施ETC電子收費，為全球首創全線無柵欄收費系統。"
    ))
    records.append(qa(
        "雪山隧道在台灣交通建設中的意義？",
        "雪山隧道全長12.9公里，為台灣最長、東南亞最長公路隧道，是國道五號的關鍵工程。連接台北與宜蘭，行車時間從2小時（北宜公路）縮短至約30分鐘。工程歷時16年（1991-2006），克服斷層、湧水等困難。通車後大幅帶動宜蘭觀光與區域發展。"
    ))
    records.append(qa(
        "What is Taiwan's highway electronic toll collection system?",
        "Taiwan's ETC system is the world's first free-flow, no-barrier tolling system. Vehicles use eTag stickers scanned at gantries along highways. Tolls are distance-based and deducted from prepaid accounts. The system handles over 14 million daily transactions with 99.9% accuracy and has been exported to India and Brazil."
    ))

    # -- Airports --
    records.append(qa(
        "桃園國際機場的運量與擴建計畫？",
        "桃園國際機場（TPE）為台灣最大國際機場，2024年旅客量約4,500萬人次。現有兩個航廈，第三航廈預計2030年完工，屆時年容量可達8,200萬人次。主要航空公司包括中華航空與長榮航空。機場捷運（2017年通車）連結台北車站與機場。"
    ))
    records.append(qa(
        "台北松山機場與高雄小港機場的定位區別？",
        "松山機場（TSA）位於台北市中心，主要服務國內航線與兩岸航線（上海、廈門等），以及部分國際商務航線（東京羽田、首爾金浦）。高雄小港機場（KHH）為台灣第二大國際機場，服務南台灣，航線包括東北亞、東南亞與兩岸航線。"
    ))

    # -- Ports --
    records.append(qa(
        "高雄港在全球航運的地位如何？",
        "高雄港是台灣最大國際商港，2024年貨櫃吞吐量約950萬TEU，全球排名第18位。擁有天然深水良港，最大可停靠22,000 TEU級貨櫃船。2019年啟用的第七貨櫃中心為全自動化智慧碼頭，由長榮海運營運。港區設有高雄自由貿易港區與高雄造船廠。"
    ))
    records.append(qa(
        "台灣還有哪些重要港口？",
        "除高雄港外：基隆港（北部主要貨櫃港，年300萬TEU）、台中港（散雜貨與穀物進口，離岸風電母港）、台北港（輔助基隆港）、花蓮港（東部散雜貨港）。這些港口正在推行智慧化與綠能轉型。"
    ))

    # -- YouBike --
    records.append(qa(
        "YouBike在台灣的發展與使用狀況？",
        "YouBike（微笑單車）2009年首發於台北，現已擴展至全台各縣市。全台總站點超過12,000站，車輛約10萬輛，日均使用量超過100萬人次。YouBike 2.0採用輕樁設計增加了設站彈性。前30分鐘免費或補貼政策鼓勵短程接駁。"
    ))
    records.append(qa(
        "What is the YouBike system in Taiwan?",
        "YouBike is Taiwan's public bicycle-sharing system, launched in Taipei in 2009. With over 12,000 stations and 100,000 bikes nationwide, it has become essential for last-mile transportation. The 2.0 version features GPS-equipped bikes with simpler docking. It's integrated with EasyCard and iPASS for seamless payment, with first 30 minutes often subsidized."
    ))

    # -- Transportation Cards --
    records.append(qa(
        "台灣的交通票證系統有哪些？悠遊卡的功能有哪些？",
        "主要電子票證：悠遊卡（EasyCard，大台北）、一卡通（iPASS，高雄/南部）、愛金卡（icash）。悠遊卡可用於台北高雄捷運、全台公車、台鐵、YouBike、停車費及便利商店小額消費。支援手機NFC支付（悠遊付）。2023年TPASS通勤月票整合各票證，提供北中南三大生活圈定額月票方案。"
    ))
    records.append(qa(
        "台灣的通勤月票TPASS是什麼？",
        "TPASS行政院通勤月票2023年7月推出。三大生活圈方案：北北基桃（NT$1,200/月，含捷運、公車、台鐵、YouBike、國道客運）、中彰投苗（NT$999/月）、南高屏（NT$999/月）。使用悠遊卡或一卡通設定，有效期內無限次搭乘指定區域公共運輸。大幅提升公共運輸使用率。"
    ))

    # =====================================================================
    # Section 5: Taiwan Food & Night Markets (~50 records)
    # =====================================================================

    # -- Taipei --
    records.append(qa(
        "台北的牛肉麵有什麼歷史和特色？",
        "台北牛肉麵是台灣最具代表性的麵食之一，每年舉辦國際牛肉麵節。紅燒以豆瓣醬、醬油、八角熬煮濃郁湯頭；清燉以牛骨與蔬菜熬製清澈湯底。知名店家：林東芳牛肉麵、永康牛肉麵、廖家牛肉麵。牛肉麵文化融合了外省老兵（1949年後遷台）與本土口味的結合。"
    ))
    records.append(qa(
        "小籠包在台灣的發展？鼎泰豐為何聞名全球？",
        "鼎泰豐創立於1958年（原食用油店），1972年轉型小籠包專賣。特色：18摺黃金比例、皮薄餡多湯汁飽滿、每顆標準21公克。全球超過170家分店（日本、美國、中國、東南亞、中東等），連續獲米其林一星。台北永康街總店為國際觀光客必訪。"
    ))
    records.append(qa(
        "What is Taiwan's beef noodle soup?",
        "Taiwanese beef noodle soup (牛肉麵) features braised beef in rich broth served with wheat noodles. Two main styles: red-braised (紅燒, using soy sauce and star anise) and clear-broth (清燉). Taipei hosts the annual International Beef Noodle Festival. Famous shops like Lin Dong Fang have achieved iconic status."
    ))
    records.append(qa(
        "台北的刈包有什麼特色？",
        "刈包（割包/虎咬豬）是台灣經典漢堡式街頭小吃。扁平發酵麵皮夾入滷五花肉、酸菜、花生粉與香菜。麵皮柔軟，五花肉滷至入口即化，花生粉與酸菜提供多層次口感。台北公館藍家刈包最知名，可選肥肉比例。"
    ))
    records.append(qa(
        "台北有什麼特色早餐？",
        "台北早餐文化豐富多樣。經典選擇包括：豆漿配油條/燒餅（阜杭豆漿、世界豆漿大王聞名）、鹹豆漿、飯糰（糯米包油條、肉鬆、酸菜）、蘿蔔糕加蛋、蛋餅（河粉蛋餅、酥皮蛋餅）。永和世界豆漿大王為24小時經營的早餐名店。"
    ))
    records.append(qa(
        "台北的芒果冰有什麼故事？",
        "芒果冰（芒果刨冰）是台灣夏季代表性冰品。永康街的芒果冰始祖冰館（Ice Monster前身）於1995年發明，將新鮮芒果搭配剉冰與煉乳。Ice Monster曾獲CNN評選為全球最佳甜點之一。台灣使用愛文芒果（台南玉井產），香氣濃郁甜度高。"
    ))

    # -- Taichung --
    records.append(qa(
        "台中的太陽餅有什麼歷史？",
        "太陽餅是台中代表性伴手禮，起源於19世紀末的台灣傳統酥餅。多層次酥皮包裹麥芽糖內餡，入口即化、甜而不膩。知名店家包括太陽堂老店（自由路，1954年創立）、阿明師老店。台中市政府每年舉辦太陽餅文化節。"
    ))
    records.append(qa(
        "台中的雞腳凍與東海蓮心冰有什麼故事？",
        "雞腳凍是東海大學商圈著名的下酒菜，以特製滷汁滷煮雞腳後去骨冰鎮，口感Q彈冰涼。東海蓮心冰以彎豆冰（彎豆+冰淇淋+煉乳）聞名。兩者皆發源於東海別墅商圈，是東海大學學生的共同記憶，已發展為宅配美食。"
    ))
    records.append(qa(
        "台中的麻薏湯是什麼？",
        "麻薏湯是台中地區獨特夏季消暑湯品，以黃麻嫩葉（麻薏）加入地瓜與小魚乾熬煮。具有清熱解毒、利尿消腫效果。味道微苦回甘，類似苦瓜但更清爽，多在夏天傳統市場中可找到。"
    ))
    records.append(qa(
        "台中美食有哪些必吃推薦？",
        "台中被稱為台灣美食之都之一，必吃美食：太陽餅（伴手禮榜首）、大麵羹（獨特鹼味黃麵，台中特有）、東泉辣椒醬搭配炒麵/米腸（台中庶民早餐經典）、麻薏湯（夏季限定）、雞腳凍（東海商圈）、豐原廟東清水排骨麵。台中也是珍珠奶茶發源地之一（春水堂）。"
    ))

    # -- Tainan --
    records.append(qa(
        "台南擔仔麵的歷史由來？",
        "擔仔麵起源於1895年，由漁民洪芋頭在日治時期渡船口挑擔叫賣。特色：使用油麵、蝦頭熬製濃郁湯頭、肉臊、鮮蝦與滷蛋。份量較小（吃巧不吃飽）。度小月擔仔麵創立於1895年，招牌繪有挑擔人物圖案，已拓展為國際連鎖品牌。"
    ))
    records.append(qa(
        "棺材板是什麼食物？",
        "棺材板是台南獨創小吃，由沙卡里巴的許六一於1942年發明。厚片土司炸至金黃酥脆，挖空填入白醬/奶油醬調製的內餡（原雞肝，現多為雞肉、蝦仁、蔬菜），蓋上土司蓋後外型似棺材而得名。酥脆外皮搭配濃郁奶香內餡為特色。"
    ))
    records.append(qa(
        "台南有哪些著名的傳統小吃？",
        "台南被稱為台灣美食之都。著名小吃：擔仔麵（度小月）、牛肉湯（現宰溫體牛清燙）、虱目魚肚湯、碗粿、蝦捲（周氏蝦捲）、鱔魚意麵、冬瓜茶（義豐阿川冬瓜茶）、米糕、棺材板。台南小吃以鮮甜口味與豐富歷史著稱。"
    ))
    records.append(qa(
        "What is Tainan's culinary significance in Taiwan?",
        "Tainan is celebrated as Taiwan's culinary capital with a 300-year food culture. The city is famous for its sweet flavor profile and extraordinary snack variety. Must-try dishes: danzai noodles (擔仔麵), coffin bread (棺材板), beef soup, milkfish soup, shrimp rolls, eel noodles, and bowl rice pudding (碗粿). Many iconic Taiwanese snacks originated here."
    ))
    records.append(qa(
        "台南的虱目魚料理有什麼特色？",
        "虱目魚（Milkfish）是台南最具代表性的養殖魚類，養殖歷史超過300年。虱目魚全身都可利用：虱目魚肚湯（油脂豐富）、虱目魚粥、魚皮湯、魚腸（乾煎魚腸為老饕最愛）、魚丸。台南虱目魚以無刺處理技術聞名。每年舉辦虱目魚文化節推廣在地產業。"
    ))

    # -- Kaohsiung --
    records.append(qa(
        "高雄的海鮮文化有什麼特色？",
        "高雄是台灣最大漁港城市，擁有前鎮漁港、蚵仔寮漁港與興達港。海鮮以新鮮平價著稱：旗魚生魚片、烏魚子（冬季特產，每對可達NT$3,000以上）、烤小卷、蚵仔煎、螃蟹粥。漁港直銷中心可現撈現烹。茄萣興達港以現切生魚片聞名。"
    ))
    records.append(qa(
        "台灣的珍珠奶茶源自何處？",
        "珍珠奶茶起源於1980年代的台中與台南之間。最主流的說法為台中春水堂（1983年）創辦人劉漢介將泡沫紅茶加入粉圓發明。另一說為台南翰林茶館（1986年）。珍珠奶茶風靡全球，在美國、日本、歐洲深受歡迎。台灣珍珠奶茶相關產業年產值超過500億新台幣。"
    ))
    records.append(qa(
        "Where did bubble tea originate?",
        "Bubble tea (boba) originated in Taiwan in the 1980s. The two most credited inventors are Chun Shui Tang in Taichung, where Liu Han-chieh added tapioca pearls to iced milk tea in 1983, and Hanlin Tea Room in Tainan. The drink has become a global phenomenon with chains like 50嵐, CoCo, Gong Cha operating worldwide."
    ))

    # -- Hualien --
    records.append(qa(
        "花蓮的麻糬有哪些特色？",
        "花蓮麻糬是東台灣最具代表性伴手禮。使用花蓮在地種植糯米手工搗製，外皮Q軟有彈性。傳統內餡為紅豆、花生與芝麻，近年發展多種口味。知名品牌：曾記麻糬（1948年創立）、阿美麻糬（小米麻糬）。花蓮每年舉辦麻糬節。"
    ))
    records.append(qa(
        "花蓮東大門夜市的特色原住民美食有哪些？",
        "花蓮東大門夜市為東台灣最大夜市。原住民美食：馬告香腸（山胡椒為原住民傳統香料）、烤飛魚（阿美族傳統）、竹筒飯、阿里鳳鳳（檳榔葉包裹月桃飯包）、樹豆豬腳湯、原住民野菜（過貓、龍鬚菜、山蘇）。夜市也有各類燒烤海鮮。"
    ))

    # -- Night Markets --
    records.append(qa(
        "士林夜市的歷史與必吃美食？",
        "士林夜市是台北規模最大的觀光夜市，歷史追溯至日治時期。分布在文林路、大東路與基河路一帶。必吃美食：豪大大雞排（比臉大雞排）、士林大香腸、生炒花枝羹、蚵仔煎、臭豆腐、大餅包小餅、青蛙下蛋。每日吸引大量國內外觀光客。"
    ))
    records.append(qa(
        "饒河夜市的歷史與推薦美食？",
        "饒河夜市位於松山區饒河街，1987年成立，是台北第一條觀光夜市。全長約600公尺，約150個攤位。必吃：福州世祖胡椒餅（炭烤，外酥內多汁，排隊名店）、藥燉排骨、麻油雞、臭豆腐、滷味。夜市連接松山慈祐宮，增添宗教文化氛圍。"
    ))
    records.append(qa(
        "寧夏夜市的特色與必吃美食？",
        "寧夏夜市位於大同區寧夏路，全長約300公尺。以千歲宴聞名（一次品嘗20多道老店小吃）。必吃：圓環邊蚵仔煎、知高飯（滷豬蹄膀飯）、豬肝榮仔湯、胡記米粉湯。寧夏夜市推動環保（全面使用環保餐具）與行動支付，獲經濟部五星級夜市認證。"
    ))
    records.append(qa(
        "逢甲夜市的特色是什麼？為什麼是大學生最愛？",
        "逢甲夜市位於台中逢甲大學周邊，是台灣最大、最創新的夜市。以創新小吃聞名，許多新興食物從此發跡。必吃：大腸包小腸（烤糯米腸夾香腸）、明倫蛋餅、熊掌包（造型割包）、官芝霖大腸包小腸。攤商不斷推陳出新，走在街頭小吃潮流最前線。"
    ))
    records.append(qa(
        "六合夜市的歷史與推薦美食？",
        "六合夜市位於高雄六合二路，1950年代形成，全長約380公尺，約170個攤位。必吃：鄭老牌木瓜牛奶（創立於1965年，多位總統造訪）、烏魚腱（台灣獨有）、海產粥、土魠魚羹、烤黑輪、鹽蒸蝦。六合夜市海產豐富，反映高雄港都飲食文化。"
    ))
    records.append(qa(
        "What are the must-visit night markets in Taiwan?",
        "Taiwan's most famous night markets: Shilin (Taipei, largest, giant chicken cutlet); Raohe (Taipei, pepper buns); Ningxia (Taipei, traditional snacks, Thousand Year Feast); Fengchia (Taichung, most innovative); Liuhe (Kaohsiung, seafood); Dongdamen (Hualien, indigenous cuisine). Each has unique character and signature dishes."
    ))
    records.append(qa(
        "台灣夜市文化有什麼獨特之處？",
        "台灣夜市文化是庶民生活的重要一環。夜市不僅是吃美食的地方，也是購物（衣服、飾品、小物）、娛樂（遊戲攤位、彈珠台、套圈圈）與社交的場所。夜市小吃價格平實、選擇多樣，從蚵仔煎到異國料理應有盡有。全台有超過300個大小夜市，每個縣市都有代表性夜市。"
    ))

    # -- Tea Culture --
    records.append(qa(
        "台灣的茶文化有哪些特色？",
        "台灣茶文化融合中國傳統製茶工藝與在地創新，以烏龍茶最著名。主要茶產區：坪林包種茶（清香型烏龍）、凍頂烏龍茶（中發酵烘焙香）、高山茶（阿里山、梨山、大禹嶺等海拔1,000公尺以上）、東方美人茶（蜜香茶）、紅玉紅茶（台茶18號，薄荷肉桂香）。台灣茶藝講究泡茶流程，木柵貓空與迪化街為品茶熱點。"
    ))
    records.append(qa(
        "What is Taiwanese oolong tea?",
        "Taiwanese oolong tea is world-renowned for high quality and diverse flavors. Key varieties: Dong Ding Oolong (medium-roasted, nutty), High Mountain Oolong (creamy floral notes from 1000m+ elevation), Oriental Beauty (heavily oxidized, honey sweetness from insect-bitten leaves), and Tieguanyin (bold roasted). Taiwan's tea culture includes traditional gongfu ceremonies."
    ))

    # -- Street Food Culture --
    records.append(qa(
        "台灣的臭豆腐為什麼受歡迎？",
        "臭豆腐以豆腐經特製滷水（含莧菜、芥菜等發酵蔬菜）浸泡發酵，產生獨特濃烈氣味。主要有三種吃法：炸臭豆腐（最普遍，配泡菜與醬油膏）、麻辣臭豆腐（加鴨血、大腸）、清蒸臭豆腐。氣味強烈但入口外酥內軟、香辣可口，是夜市經典美食。"
    ))
    records.append(qa(
        "台灣的滷味文化是什麼？",
        "滷味以特製滷汁（醬油、冰糖、八角、桂皮、花椒等藥材）長時間滷煮各類食材。分為加熱滷味（現煮熱食）、冰鎮滷味（冷盤）、乾滷味（水分收乾如鐵蛋）。食材從雞腳、豬耳朵、豆干、海帶到科學麵、高麗菜皆可滷。師大夜市、公館等地有許多知名滷味攤。"
    ))
    records.append(qa(
        "台灣的雞排文化是如何發展的？",
        "台灣雞排是1990年代興起的街頭小吃。去骨雞胸肉拍平後以醬油、蒜泥、糖等醃漬裹粉油炸。士林豪大大雞排以比臉大尺寸打響名號。發展出脆皮雞排、起司雞排、辣味雞排等變化版。台灣每年消費超過2億片雞排。"
    ))
    records.append(qa(
        "台灣的火鍋文化有什麼特色？",
        "台灣火鍋多元豐富，主要分為：麻辣鍋（鼎王、老四川）、吃到飽火鍋（千葉、馬辣）、個人涮涮鍋、羊肉爐、薑母鴨（冬季進補）、酸菜白肉鍋。台灣火鍋強調湯頭與沾醬（沙茶醬為經典），食材從海鮮到手工餃類應有盡有。火鍋店全年無休，夏天冷氣照樣營運。"
    ))

    # -- Food Safety --
    records.append(qa(
        "台灣的食品安全管理制度如何？",
        "台灣食品安全由衛福部TFDA統籌監管。主要制度：HACCP強制導入、食品追溯追蹤系統、食品添加物正面表列、CAS優良農產品標章。2011年塑化劑與2014年餿水油事件後修訂《食品安全衛生管理法》，大幅提高罰則與檢驗力度。"
    ))
    records.append(qa(
        "台灣有哪些重要的食品安全事件？",
        "重要食品安全事件：2011年塑化劑污染（DEHP污染飲料與保健食品）、2013年毒澱粉（順丁烯二酸酐化製澱粉）、2014年餿水油（強冠回收廚餘提煉劣質豬油）。這些事件促使政府改革食安制度，建立食品雲追溯系統與三級品管制度。"
    ))

    # -- International --
    records.append(qa(
        "台灣米的種類與米食文化？",
        "台灣主要栽種蓬萊米（稉稻）與在來米（秈稻）。蓬萊米適合白飯與壽司；在來米用於蘿蔔糕、粄條、米苔目。特色米製品：碗粿（南台灣）、米糕（台南米糕與筒仔米糕）、客家粄條（美濃）、米粉（新竹米粉）、油飯。台灣產越光米（台南16號）口感極佳。"
    ))
    records.append(qa(
        "What is Taiwanese bento culture like?",
        "Taiwanese bento (便當) is a ubiquitous lunch culture. The iconic Railway Bento (台鐵便當) features braised pork chop, egg, tofu, pickled vegetables, and rice in a tin box for about NT$100. Convenience stores offer enormous variety. Bento represents Taiwan's practical food heritage: balanced, portable, and satisfying."
    ))
    records.append(qa(
        "台灣的異國料理有哪些？",
        "台灣的異國料理豐富多元。日式料理（拉麵、壽司、丼飯）最為普及；韓式（韓式炸雞、石鍋拌飯）持續熱門；泰式（打拋豬、綠咖哩）與越式（河粉、春捲）廣受歡迎。台北天母與台中精明街有美式/歐式餐廳聚集。移民帶入的滇緬（雲南）、印尼與菲律賓料理也在各地形成特色聚落。"
    ))

    # =====================================================================
    # Section 6: Taiwan Nature & Environment (~30 records)
    # =====================================================================

    # -- National Parks --
    records.append(qa(
        "台灣有幾個國家公園？分別的特色是什麼？",
        "台灣共有9個國家公園：墾丁（海洋生態與珊瑚礁）、玉山（台灣最高峰3,952公尺）、陽明山（火山地質與溫泉）、太魯閣（大理石峽谷）、雪霸（雪山與大霸尖山，冰河遺跡）、金門（戰役史蹟與閩南文化）、東沙環礁（海洋保護區）、台江（濕地與紅樹林）、澎湖南方四島（玄武岩與珊瑚礁）。面積總計約佔台灣陸域8.6%。"
    ))
    records.append(qa(
        "太魯閣國家公園的地理與地質特色？",
        "太魯閣國家公園位於花蓮，以世界級大理石峽谷景觀聞名。立霧溪切割碳酸鈣大理石岩層形成垂直陡峭峽谷，最深達1,000公尺以上。著名景點：長春祠、燕子口、九曲洞、慈母橋、天祥與白楊步道。中橫公路穿越峽谷。2024年4月3日花蓮大地震對太魯閣造成嚴重破壞。"
    ))
    records.append(qa(
        "玉山國家公園的特色與登山資訊？",
        "玉山（3,952公尺）為東亞最高峰之一。玉山國家公園面積超10萬公頃。攀登主峰從塔塔加鞍部起登，全長10.9公里，需兩天一夜（排雲山莊住宿）。每日限額約100人，需申請入園許可與入山證。玉山群峰保留完整原始森林、高山草原與多樣野生動物。"
    ))
    records.append(qa(
        "What is Taroko Gorge and why is it famous?",
        "Taroko Gorge (太魯閣) is a spectacular marble canyon in eastern Taiwan, carved by the Liwu River over millions of years. The gorge features vertical marble walls up to 1,000m high. Key attractions: Swallow Grotto, Tunnel of Nine Turns, Eternal Spring Shrine. The Central Cross-Island Highway traverses the gorge with tunnels carved directly into cliffs."
    ))
    records.append(qa(
        "墾丁國家公園的海洋生態有什麼特色？",
        "墾丁國家公園位於台灣最南端，擁有豐富海洋生態與珊瑚礁生態系。珊瑚礁覆蓋率曾達80%以上，有300種以上珊瑚與1,500種以上魚類。著名潛水點：後壁湖、出水口、帆船石。每年4月珊瑚產卵為年度生態盛事。墾丁也是台灣唯一熱帶海域。"
    ))
    records.append(qa(
        "陽明山國家公園有什麼自然與人文特色？",
        "陽明山國家公園位於台北北郊，是台灣唯一擁有火山地質的國家公園。大屯火山群包含七星山（1,120公尺）、大屯山等，擁有硫磺噴氣孔（小油坑）與溫泉。春季櫻花、初夏繡球花、秋季芒草為四季花景。著名景點：竹子湖（海芋）、擎天崗（草原）。"
    ))
    records.append(qa(
        "台灣有哪些國家森林遊樂區推薦？",
        "台灣有超過20個國家森林遊樂區，推薦：太平山（宜蘭，蹦蹦車與檜木森林）、阿里山（嘉義，日出雲海與神木鐵路）、大雪山（台中，中海拔雲海與鳥類）、奧萬大（南投，秋楓）、池南（花蓮，生態池）、知本（台東，溫泉森林）。各園區提供步道、自然教育中心與住宿設施。"
    ))

    # -- Endangered Species --
    records.append(qa(
        "台灣黑熊的保育現狀如何？",
        "台灣黑熊（Formosan Black Bear）是台灣唯一原生熊類，瀕臨絕種。野外數量僅200-600隻，分布於中央山脈與雪山山脈。保育措施：設立保護廊道、推廣防熊圍籬、加強查緝盜獵。黑熊媽媽黃美秀教授團隊成功野放救傷黑熊，為生態保育典範。"
    ))
    records.append(qa(
        "台灣雲豹是否已經滅絕？",
        "台灣雲豹（Formosan Clouded Leopard）為台灣最大貓科動物，台灣特有亞種。自1980年代後無確切野外觀察記錄，學術界普遍認為可能已滅絕。台灣雲豹在原住民文化中具重要地位（排灣族與魯凱族視為聖獸）。生物多樣性研究所持續進行自動相機監測。"
    ))
    records.append(qa(
        "What endangered species are found in Taiwan?",
        "Endangered endemic species in Taiwan: Formosan Black Bear (200-600 remaining), Formosan Pangolin (critically endangered), Formosan Landlocked Salmon (only in Shei-Pa National Park), Formosan Sika Deer (reintroduced after extinction in the wild), Formosan Rock Macaque (Taiwan's only native primate, threatened by habitat loss)."
    ))
    records.append(qa(
        "台灣的櫻花鉤吻鮭有什麼保育故事？",
        "櫻花鉤吻鮭（Formosan Landlocked Salmon）為台灣特有亞種，僅分布於雪霸國家公園七家灣溪流域。冰河時期孑遺物種，為全球分布最南端鮭魚。1990年代數量一度降至200尾。復育計畫包括棲地改善、人工繁殖放流、水質監測。2024年族群回升至約7,000-9,000尾。"
    ))

    # -- Environmental Issues --
    records.append(qa(
        "台灣的空氣污染問題主要來自哪些來源？",
        "台灣空氣污染來源：工業排放（台中火力發電廠、中鋼等）、交通運輸（柴油車與機車廢氣）、境外傳輸（中國大陸霧霾，冬季東北季風時最明顯）。主要污染物為PM2.5，南部空氣品質通常較北部差。政府推動老舊車輛汰換、工廠排放加嚴、燃煤電廠降載等改善措施。"
    ))
    records.append(qa(
        "台灣的垃圾分類與回收制度有什麼特色？",
        "台灣垃圾分類與回收制度聞名全球，回收率超過60%。1990年代推動垃圾不落地（定時定點收運），2005年實施強制分類。垃圾分為一般垃圾、資源回收物與廚餘三大類。隨袋徵收（新北市）或指定專用垃圾袋（台北市）達到垃圾減量。台灣廢棄物管理成就吸引各國取經。"
    ))
    records.append(qa(
        "What is Taiwan's waste management system like?",
        "Taiwan's waste management achieves a recycling rate of over 60%. The 1990s 'Keep Trash Off the Ground' policy required scheduled waste disposal. The 2005 mandatory recycling separates waste into three categories: general, recyclables, and food waste. Taipei's pay-per-bag system uses government-issued bags for general waste while recycling is free."
    ))

    # -- Conservation --
    records.append(qa(
        "台灣的海洋保育有哪些重要措施？",
        "海洋保育措施包括：設立海洋委員會（2018年）、劃設海洋保護區（約8%領海）、實施漁獲配額與休漁期、禁止捕撈鯨鯊與鯨豚。2020年通過《海洋基本法》。近年推動珊瑚礁復育、海龜保育（小琉球綠蠵龜密度全台最高）與海洋垃圾清除。"
    ))
    records.append(qa(
        "台灣的森林覆蓋率與林業政策？",
        "台灣森林覆蓋率約60%，為東亞最高之一。1990年代起全面停止天然林商業採伐，轉向保育與國土保安。主要林型：熱帶季風林（南部低海拔）、亞熱帶闊葉林（中低海拔）、針葉林（中高海拔）、高山苔原（3,500公尺以上）。林業署推動國家森林遊樂區與自然步道系統。"
    ))
    records.append(qa(
        "台灣的濕地保育有哪些重要成就？",
        "台灣有82處國家重要濕地。國際級：台江國家公園四草濕地、彰化海岸濕地。重要成就：台江國家公園（首個濕地型國家公園）、高美濕地（復育成功案例）。黑面琵鷺在台度冬數量從1990年代的300隻增至2024年的約4,000隻，為保育重大成果。"
    ))

    # -- Hot Springs --
    records.append(qa(
        "台灣的溫泉文化有什麼特色？",
        "台灣位於環太平洋火山帶，地熱資源豐富。主要溫泉區：北投（白磺泉，日治時期開發）、烏來（碳酸氫鈉泉，泰雅族文化）、關子嶺（台南，泥漿溫泉世界罕見）、知本（台東，鹼性碳酸泉）、礁溪（宜蘭，平地溫泉）。台灣溫泉文化融合日治溫泉旅館傳統與現代SPA養生。"
    ))
    records.append(qa(
        "What are Taiwan's best hot spring destinations?",
        "Beitou (Taipei) offers white sulfur springs with historic Japanese bathhouses. Wulai has carbonated springs with indigenous Atayal culture. Guanziling (Tainan) features unique mud springs, the only such springs in Southeast Asia. Jiaoxi (Yilan) has odorless, clear springs perfect for families. Zhiben (Taitung) offers mountain relaxation."
    ))

    # -- Whale Watching --
    records.append(qa(
        "台灣的賞鯨活動集中在哪些地方？",
        "賞鯨以東海岸為主要區域，花蓮與宜蘭為最主要出發點。季節為每年4月至10月。常見物種：瓶鼻海豚、飛旋海豚、熱帶斑海豚、花紋海豚，以及抹香鯨、虎鯨（稀少）、大翅鯨（冬春季稀有訪客）。宜蘭龜山島海域亦為重要賞鯨地點。賞鯨成功率通常在90%以上。"
    ))
    records.append(qa(
        "台灣的海洋生態多樣性如何？",
        "台灣周邊海域記錄超過1,200種魚類、300種珊瑚、600種甲殼類與500種軟體動物。台灣位於黑潮（暖流）與中國沿岸流（寒流）交匯處，生態多樣性達全球平均4-5倍。綠島、蘭嶼與小琉球為著名海洋生態旅遊地點，以小琉球綠蠵龜密度最高。"
    ))

    # -- Hiking --
    records.append(qa(
        "台灣有哪些著名的登山步道？",
        "台灣登山資源豐富，從郊山到高山百岳應有盡有。著名步道：台北象山步道（101景觀）、陽明山東西大縱走、草嶺古道（秋季芒花季）、錐麓古道（太魯閣懸崖步道）、玉山主峰步道、雪山主東峰步道（冰河圈谷）、嘉明湖步道（天使的眼淚）、阿朗壹古道（海岸原始步道）。台灣百岳為海拔3,000公尺以上100座高山。"
    ))
    records.append(qa(
        "What are Taiwan's most popular hiking trails?",
        "Taiwan offers world-class hiking: Elephant Mountain for Taipei skyline; Yangmingshan Seven-Star Mountain; Zhuilu Old Trail (cliffside); Yushan Main Peak (3,952m, Taiwan's highest); Shei-Pa trails with glacial cirques; Jiaming Lake (alpine scenery). Taiwan has 100 'Peaks' over 3,000m known as the Baiyue (百岳), a serious hiker's challenge."
    ))
    records.append(qa(
        "台灣的自行車旅遊路線有哪些推薦？",
        "台灣自行車旅遊蓬勃發展。推薦路線：環島1號線（全長約1,000公里，為期9-12天環島騎行）、日月潭環潭自行車道（CNN評選全球十大美景單車道）、池上伯朗大道（金城武樹）、東豐自行車綠廊（后豐鐵馬道，舊山線隧道）。台灣推廣自行車環島文化，每年有數萬人完成環島。"
    ))
    records.append(qa(
        "台灣的離島有哪些值得旅遊的？",
        "台灣離島各具特色：澎湖群島（玄武岩地質、花火節、跨海大橋、仙人掌冰）、綠島（海底溫泉、潛水天堂、監獄遺址）、蘭嶼（達悟族文化、拼板舟、飛魚季）、小琉球（綠蠵龜生態、珊瑚礁）、馬祖（藍眼淚、芹壁聚落、戰地遺跡）、金門（閩南古厝、高粱酒、戰史文化）。近年離島旅遊熱度持續上升。"
    ))

    # Additional Politics & Government records
    records.append(qa(
        "中華民國國旗的由來與意義？",
        "中華民國國旗（青天白日滿地紅）由孫中山先生設計，1928年正式採用。青色代表光明磊落與青天，白色代表清廉正直與白日，紅色代表革命犧牲與熱血。十二道光芒代表一天12時辰、一年12個月，象徵自由平等博愛的精神。國旗與國歌《三民主義歌》同在升降旗儀式中使用。"
    ))
    records.append(qa(
        "台灣的政府預算如何編列與審查？",
        "台灣政府預算由行政院主計總處編列，經行政院會通過後送立法院審議。立法院進行朝野協商與逐項表決，三讀通過後才能動支。預算會計年度為每年1月1日至12月31日。近年預算爭議焦點包括國防預算、社會福利支出與能源補貼等。審計部（監察院所屬）負責決算審核。"
    ))
    records.append(qa(
        "台灣的媒體環境與新聞自由現狀？",
        "台灣的新聞自由在亞洲名列前茅，2024年無國界記者組織（RSF）新聞自由指數台灣排名全球第27名。主要媒體包括公共電視（PTS）、三立電視、TVBS、民視、中國時報、聯合報、自由時報等。媒體面臨的挑戰包括假資訊（misinformation）泛濫、媒體立場極化、以及社群媒體對傳統媒體的衝擊。"
    ))
    records.append(qa(
        "What is the role of the Control Yuan in Taiwan?",
        "The Control Yuan (監察院) is Taiwan's oversight branch, uniquely positioned in the Five-Power Constitution. It exercises impeachment, censure, and audit powers over government officials. Composed of 29 members nominated by the President and confirmed by the Legislative Yuan, the Control Yuan investigates government misconduct and audits financial accounts through the Ministry of Audit."
    ))
    records.append(qa(
        "台灣的憲法法庭是什麼？",
        "憲法法庭是司法院的一部分，2019年《憲法訴訟法》施行後取代大法官會議制度。憲法法庭由15位大法官組成，審理法規範憲法審查、裁判憲法審查、機關爭議、總統副總統彈劾、政黨違憲解散及地方自治保障等案件。2022年憲法法庭做出重要判決，包括死刑制度合憲性解釋等。"
    ))

    # Additional Economy records
    records.append(qa(
        "台灣的外資投資環境如何？",
        "台灣外資投資環境以高科技製造業為最強吸引力。2023年台灣核准外人投資約120億美元，主要來源為荷蘭（多為半導體相關）、日本、美國與德國。台灣的優勢包括：完整的供應鏈、高素質人才、智慧財產權保護完善（連續多年列美國特別301觀察名單外）、以及穩定的總體經濟。劣勢包括內需市場較小與地緣政治風險。"
    ))
    records.append(qa(
        "台灣的數位經濟規模有多大？",
        "台灣數位經濟佔GDP比重約25-30%，高於全球平均。電子商務市場規模約3,500億新台幣，momo購物網為台灣最大電商平台。行動支付普及率在政府推廣下從2019年的40%成長至2024年的80%以上。LINE Pay、街口支付、玉山Wallet等為主要支付工具。"
    ))
    records.append(qa(
        "What is Taiwan's tax system like?",
        "Taiwan's tax system includes individual income tax (progressive, rates 5-40%), business income tax (17%), value-added tax (5%, the lowest among developed Asian economies), and estate/gift tax. The individual income tax system features a standard deduction of NT$124,000 and a special salary deduction of NT$207,000 as of 2024. Capital gains from securities trading are tax-free, a unique feature that has boosted stock market participation."
    ))
    records.append(qa(
        "台灣的農業現狀與挑戰？",
        "台灣農業佔GDP約1.8%，就業人口約5%。主要農產品：稻米、蔬菜、水果（芒果、鳳梨、香蕉、蓮霧、釋迦）、茶葉、豬肉、漁產品。台灣農業以小農為主，面臨人口老化（農民平均年齡超過62歲）、農業用水競爭、自由貿易開放壓力（如加入CPTPP）、以及極端氣候影響。政府推動智慧農業、青年返鄉從農補助與農產品出口拓銷。"
    ))
    records.append(qa(
        "台灣的能源進口依賴度有多高？",
        "台灣能源進口依賴度高達97%以上，為能源安全重大挑戰。主要進口能源：原油（主要來自中東）、煤礦（主要來自澳洲與印尼）、天然氣（主要來自卡達、澳洲與美國）。2024年台灣能源進口總額約500億美元。推動再生能源（太陽能、離岸風電）被視為降低能源進口依賴的關鍵策略。"
    ))
    records.append(qa(
        "台灣的社會福利制度有哪些？",
        "台灣社會福利制度包括：全民健康保險（1995年開辦，覆蓋率99%以上）、國民年金保險（2008年開辦，保障未參加其他社會保險者）、勞工保險（含生育、傷病、失能、老年給付）、就業保險（失業給付與職業訓練）、以及各類社會救助（低收入戶補助、中低收入老人生活津貼）。2024年社會福利支出約佔中央政府總預算25%。"
    ))

    # Additional Technology records
    records.append(qa(
        "台灣在量子科技方面的發展？",
        "台灣在量子科技領域積極布局。中央研究院成立量子科技研究團隊，發展量子電腦與量子通訊。台灣大學、清華大學等也設立量子研究中心。2023年國科會啟動量子國家隊計畫，投入約80億新台幣。工研院與半導體公司合作開發量子位元製程技術。台灣目標在2030年前開發出實用型量子電腦原型。"
    ))
    records.append(qa(
        "台灣的5G網路建設與應用狀況？",
        "台灣5G網路於2020年7月正式開台，由中華電信、台灣大哥大、遠傳電信等業者提供服務。2024年5G覆蓋率達95%以上，用戶滲透率約35%。主要應用包括：智慧工廠（遠距監控、自動化）、智慧醫療（遠距手術指導）、自駕車通訊、AR/VR娛樂。台灣5G資費為全球最低之一（約NT$599起）。"
    ))
    records.append(qa(
        "What is Taiwan's semiconductor equipment industry?",
        "Taiwan's semiconductor equipment industry has grown alongside its chip manufacturing dominance. Key companies: ASMPT (Singapore-based but large Taiwan operations), Disney/Lam Research and Applied Materials have major service centers in Taiwan. Homegrown equipment makers include Grand Process Technology, Marketech, and ULVAC Taiwan. The equipment service and parts market in Taiwan is estimated at over US$10 billion annually."
    ))
    records.append(qa(
        "台灣的物聯網（IoT）產業發展如何？",
        "台灣在物聯網（IoT）產業具有硬體製造優勢，晶片（聯發科、瑞昱）、感測器（原相、義隆）、通訊模組（中磊、啟碁）在全球市場佔有一席之地。工研院推動物聯網感測平台技術。台灣在智慧家庭、智慧製造、智慧農業等IoT應用領域皆有廠商布局。台灣IoT產值在2024年約5,000億新台幣。"
    ))
    records.append(qa(
        "台灣的電動車產業布局有哪些？",
        "台灣電動車產業以零組件供應鏈為核心。鴻海MIH電動車開放平台聯盟整合超過2,500家會員，推動電動車模組化製造。台達電子為全球電動車電源管理與充電樁領導廠商。貿聯-KY、胡連、同致等為Tesla等車廠供應鏈成員。台灣目標在2030年電動車市售占比達30%，2040年達100%。"
    ))
    records.append(qa(
        "台灣的生態農業與有機農業發展？",
        "台灣有機農業面積約1.2萬公頃，佔農地面積約1.5%。政府推動有機農業促進法（2019年施行），提供有機驗證補助與銷售通路輔導。特色有機產品包括：有機米（花蓮富里、台東池上）、有機茶（坪林、三峽）、有機蔬菜（主婦聯盟、里仁通路）。台灣也推動友善耕作（綠色保育標章）與生態農業，兼顧生產與生態保育。"
    ))
    records.append(qa(
        "台灣有哪些重要的軟體與SaaS公司？",
        "台灣軟體與SaaS（軟體即服務）產業近年快速成長。代表性公司：Appier（AI行銷平台，東京上市）、Trend Micro趨勢科技（全球資安軟體領導者，成立於台灣）、Kdan Mobile凱鈿（創意工具SaaS，全球破億用戶）、Gogolook（來電辨識Whoscall，證交所創新板上市）、SurveyCake（雲端問卷平台）、PicCollage（拼貼趣，全球超過2億下載）。"
    ))
    records.append(qa(
        "台灣的政府開放資料政策如何？",
        "台灣政府推動開放資料（Open Data）政策，在全球開放資料評比中名列前茅（2019年曾獲全球第一）。data.gov.tw平台提供超過5萬個政府資料集。重點資料包括：氣象、交通（公車動態、即時路況）、醫療（健保就醫統計）、環境（空氣品質監測）、經濟（貿易統計）。開放資料帶動了交通App、天氣預報、不動產查詢等民間創新應用。"
    ))
    records.append(qa(
        "台灣的半導體人才培育體系是怎樣的？",
        "台灣半導體人才培育從大學到產業緊密結合。重點大學包括：國立台灣大學（電機資訊學院）、國立清華大學（電機資訊學院，半導體學院）、國立陽明交通大學（半導體學院，與台積電合作）、成功大學（微電子研究所）、台灣科技大學。台積電設立台積電半導體學院，與大學合作開設專業課程，每年培育超過千名半導體工程師。政府也推動半導體國際學程吸引外國人才。"
    ))
    records.append(qa(
        "台灣的災害防救科技發展如何？",
        "台灣位於地震與颱風多發區，災害防救科技發達。中央氣象局提供精準的颱風路徑預報（領先全球多國）。國家地震工程研究中心（NCREE）擁有亞洲最大地震模擬振動台，進行建築抗震測試。災防科技中心開發即時災情彙整系統。1999年921大地震後，台灣大幅強化建築耐震規範、救災體系與民眾防災意識。"
    ))

    # Additional Transportation records
    records.append(qa(
        "台北車站在台灣交通樞紐的角色？",
        "台北車站是台灣最重要的交通樞紐，五鐵共構（台鐵、高鐵、捷運、機場捷運、客運轉運站）。每日進出人次超過50萬。站內設有台鐵月台、高鐵月台、台北捷運紅線與藍線，以及機場捷運乘車處。地下商店街（台北地下街、誠品站前店）形成大型購物商場。台北車站前身為1891年清代鐵路起點，現已成為台灣交通的門戶。"
    ))
    records.append(qa(
        "台灣的智慧交通系統有哪些成就？",
        "台灣智慧交通系統成就豐碩。高速公路ETC收費系統為全球首個無柵欄電子收費系統，啟用時間最久、成功率最高（99.9%以上）。各縣市公車動態資訊系統（如台北等公車App）讓乘客即時掌握公車到站時間。台北市智慧號誌系統（交通順暢計畫）根據車流即時調整紅綠燈。台灣智慧交通解決方案已輸出至東南亞與中東國家。"
    ))
    records.append(qa(
        "What is the Taiwan Tourist Shuttle service?",
        "The Taiwan Tourist Shuttle (台灣好行) is a bus service network operated by the Tourism Administration, linking major train stations and HSR stations to popular tourist attractions across the island. With over 80 routes covering most scenic spots, it offers convenient and affordable transportation for independent travelers. Routes include Alishan, Sun Moon Lake, Taroko, Kenting, and many more. Fares are typically under NT$200 per trip."
    ))
    records.append(qa(
        "台灣的公路客運與國道客運市場概況？",
        "台灣公路客運分為市區公車、公路客運（跨縣市）與國道客運（高速公路長途）。主要國道客運業者：國光客運、統聯客運、和欣客運、阿羅哈客運。台北至高雄國道客運票價約NT$350-500，行車時間約4-5小時，為高鐵之外的經濟選擇。近年面臨高鐵競爭與駕駛員短缺的挑戰，逐步轉型。"
    ))
    records.append(qa(
        "台灣的電動公車推廣狀況？",
        "台灣推動電動公車有成，2024年全台電動公車約1,800輛，約佔公車總數的15%。政府目標2030年市區公車全面電動化。主要電動公車供應商包括成運汽車、華德動能、創奕能源。電動公車以中台灣（台中、彰化）與南台灣（高雄）推廣最積極。智慧充電管理系統也同步開發中。"
    ))
    records.append(qa(
        "台灣的無障礙交通設施有哪些？",
        "台灣的無障礙交通設施持續改善。台北捷運全線設置無障礙電梯、導盲磚、輪椅專屬空間與低位服務台。台鐵與高鐵車站設有無障礙坡道與電梯，部分列車配置無障礙車廂與輪椅座位。低地板公車（低底盤公車）在全台各縣市逐步普及，方便輪椅與嬰兒車乘客。復康巴士提供預約式身心障礙者交通服務。"
    ))

    # Additional Food records
    records.append(qa(
        "台灣的蚵仔煎有什麼歷史？",
        "蚵仔煎（Oyster Omelette）是台灣最具代表性的夜市小吃之一。相傳源自荷蘭統治時期（17世紀），當時荷蘭人引入蚵仔養殖，在地居民以蚵仔混合番薯粉煎煮而成。現代作法以鮮蚵、雞蛋、青菜（小白菜或萵苣）搭配地瓜粉漿，煎至金黃後淋上甜辣醬。各家口味略有不同，以寧夏夜市圓環邊蚵仔煎、豐原廟東蚵仔煎最為知名。"
    ))
    records.append(qa(
        "碗粿是什麼樣的食物？",
        "碗粿是台灣傳統米食點心，以在來米漿加入餡料放入碗中蒸製而成。分為兩種主要流派：南部碗粿（台南最具代表性）以醬油色澤與豐富內餡（肉臊、鹹蛋黃、香菇）為特色，口感軟綿入味；北部碗粿則較為白淨，口感偏硬，通常搭配蘿蔔乾與醬油膏食用。台南富盛號碗粿為最知名的老店。"
    ))
    records.append(qa(
        "What is stinky tofu and why do people love it?",
        "Stinky tofu (臭豆腐) is a fermented tofu dish known for its strong, pungent odor. The fermentation brine includes vegetables like amaranth and mustard greens, creating a distinctive smell. Despite the challenging aroma, the taste is mild and flavorful. Deep-fried stinky tofu is the most common style, served with pickled cabbage and soy sauce. It's considered an acquired taste and a must-try adventurous food in Taiwan's night markets."
    ))
    records.append(qa(
        "台灣的三杯雞是什麼料理？",
        "三杯雞是台灣經典的客家菜與台菜料理，以一杯米酒、一杯醬油、一杯麻油（各一杯）燜煮雞肉，加入薑片、蒜頭、辣椒與九層塔（羅勒）提味。三杯料理也可應用於中卷（三杯中卷）或豆腐。三杯雞以醬香濃郁、酒香四溢為特色，為台灣餐館與家庭餐桌的常見菜色。"
    ))
    records.append(qa(
        "台灣的夜市遊戲有哪些？",
        "台灣夜市除了美食，還有豐富的遊戲攤位。經典遊戲包括：彈珠台（打彈珠換糖果或獎品）、套圈圈（套中瓶子得獎品）、射氣球（以飛鏢射破氣球計分）、撈金魚（紙網撈金魚）、麻將賓果（麻將連線遊戲）。這些遊戲是台灣夜市的獨特娛樂文化，尤其吸引家庭與年輕族群。"
    ))
    records.append(qa(
        "台灣的鳳梨酥有什麼文化意義？",
        "鳳梨酥是台灣最具代表性的伴手禮之一。鳳梨酥外皮以奶油麵粉製成酥脆餅皮，內餡以鳳梨醬（加入冬瓜醬或純鳳梨）為主。鳳梨的台語發音（ong-lai）與「旺來」同音，象徵好運與興旺，使其成為年節送禮的首選。知名品牌包括微熱山丘（使用八卦山土鳳梨）、佳德鳳梨酥、李鵠餅店（基隆）、日出（台中）。"
    ))
    records.append(qa(
        "台灣的客家菜有什麼特色？",
        "台灣客家菜以鹹、香、肥為特色，善用醃漬與發酵食材。經典客家菜包括：客家小炒（魷魚、豆乾、豬肉絲爆炒）、梅干扣肉（梅干菜與五花肉蒸製）、薑絲大腸（豬大腸與酸菜、薑絲快炒）、福菜湯（芥菜醃製發酵湯品）、擂茶（茶葉、芝麻、花生等研磨沖泡）。桃園、新竹、苗栗與高雄美濃為主要客家菜聚集地。"
    ))
    records.append(qa(
        "台灣的冰品文化有哪些特色？",
        "台灣冰品文化豐富多元。傳統冰品：剉冰（配料紅豆、綠豆、芋圓、仙草、愛玉）、雪花冰（牛奶冰磚刨製，口感綿密）、枝仔冰（傳統冰棒）。特色冰品包括：芒果冰（永康街Ice Monster）、泡泡冰（基隆廟口花生泡泡冰）、冷熱冰（屏東潮州，配料熱芋泥搭配刨冰）、嫩仙草凍（復興空廚）。台灣冰品全年無休，冬季也有熱賣的傳統甜湯（紅豆湯、花生湯）。"
    ))
    records.append(qa(
        "台灣的素食文化發展如何？",
        "台灣的素食文化深厚，全台約有12-15%人口為素食或蔬食者。台灣素食以宗教素（佛教、一貫道）為主流，也有健康素與環保素。台北有世界密度最高的素食餐廳。台灣發明素肉（植物肉）、素雞等加工技術成熟。自助餐式素食餐廳（如養生蔬食自助餐）與吃到飽素食餐廳非常普及。慈濟推廣的素食理念影響力廣泛。"
    ))
    records.append(qa(
        "What is the Taiwanese hot pot experience like?",
        "Taiwanese hot pot is a social dining experience where diners cook raw ingredients in a simmering broth at the table. Popular broth options include spicy mala (麻辣), sukiyaki (壽喜燒), tomato, and herbal chicken. The dipping sauce station is a key feature, with choices like沙茶醬 (satay sauce), garlic, chili, cilantro, and raw egg. All-you-can-eat hot pot chains like 馬辣 and 千葉 are wildly popular. Hot pot is enjoyed year-round, especially during winter."
    ))
    records.append(qa(
        "台灣的在地食材特色有哪些？",
        "台灣各地擁有豐富的在地食材：台南學甲的虱目魚、屏東的芒果與蓮霧、台東的釋迦、宜蘭的三星蔥、南投的茶葉與香菇、雲林的醬油（西螺瑞春、丸莊）、嘉義的火雞肉飯與阿里山山葵。台灣小農市集（如水花園有機農夫市集）蓬勃發展，推廣產地直送與友善環境農業。"
    ))
    records.append(qa(
        "台灣醃漬食品的文化有哪些？",
        "台灣醃漬食品文化源自早期保存食材的需求。常見醃漬食品：醬瓜（醬油醃製小黃瓜）、蔭瓜（黑豆豉醃製）、破布子（樹子醃製品）、菜脯（醃蘿蔔乾）、酸菜（芥菜發酵）、豆腐乳（豆腐發酵）、豆豉（黑豆發酵）。客家福菜與梅干菜為醃漬文化代表。這些醃漬品不僅是配菜，也是料理調味的靈魂食材。"
    ))

    # Additional Nature records
    records.append(qa(
        "台灣的珊瑚礁生態面臨哪些威脅？",
        "台灣珊瑚礁生態面臨的主要威脅包括：氣候變遷導致海水升溫（珊瑚白化，2020年台灣南部與離島發生大規模白化事件）、海洋酸化、過度漁撈與毒魚、觀光衝擊（防曬乳化學成分、潛水踩踏）、廢水污染與沉積物。墾丁、綠島與小琉球是受到關注的重點區域。政府與民間推動珊瑚復育、設立海洋保護區及推廣防曬替代措施。"
    ))
    records.append(qa(
        "台灣的候鳥遷徙路線與賞鳥地點？",
        "台灣位於東亞-澳大利亞候鳥遷徙路線的重要中繼站，每年春秋兩季有大量候鳥過境或度冬。主要賞鳥地點：關渡自然公園（台北，水鳥重要棲地）、大園許厝港（桃園，雁鴨與鷸鴴類）、彰化海岸（大杓鷸、黑面琵鷺）、曾文溪口（台南，黑面琵鷺主要度冬區）、墾丁國家公園（秋季猛禽遷徙，赤腹鷹與灰面鵟鷹過境）、馬祖（神話之鳥黑嘴端鳳頭燕鷗）。台灣記錄鳥種超過650種。"
    ))
    records.append(qa(
        "What are Taiwan's most beautiful lakes?",
        "Taiwan's most beautiful lakes include: Sun Moon Lake (日月潭) in Nantou, the largest natural lake in Taiwan, famous for its turquoise waters and surrounding tea plantations; Jiaming Lake (嘉明湖) in Taitung, a high-altitude (3,310m) glacial lake known as the 'Angel's Tear'; Chiaming Lake (翠峰湖) in Yilan, Taiwan's largest high-mountain lake; and the Four Seasons Lake (四季湖) in Nantou. These lakes offer stunning scenery and excellent hiking opportunities."
    ))
    records.append(qa(
        "台灣的地質景觀多樣性如何？",
        "台灣位於歐亞板塊與菲律賓海板塊交界帶，地質景觀極具多樣性。主要地質景觀：花東縱谷（板塊縫合線）、太魯閣大理石峽谷、野柳女王頭（蕈狀岩，海蝕地形）、澎湖玄武岩柱狀節理、東北角海岸（海蝕平台與奇岩）、關子嶺泥火山、墾丁出火（天然氣滲漏自燃）、南橫地熱與溫泉、以及全新世火山（大屯火山群）。台灣的地質豐富度在全球島嶼國家中數一數二。"
    ))
    records.append(qa(
        "台灣的瀕危植物有哪些保育成果？",
        "台灣特有植物比例極高（約4,000種維管束植物中30%為特有種）。重要的瀕危植物保育成果包括：台灣杉（棲蘭山區保存台灣最大原始巨木林）、台灣紅檜與扁柏（珍貴針葉樹，人工復育有成）、台灣油杉（冰河孑遺物種，人工扦插繁殖成功）、鐘萼木（藍綠色的台灣特有喬木）、玉山薊（高山特有植物）。植物保種中心與各植物園進行種子庫保存與復育工作。"
    ))
    records.append(qa(
        "台灣的生態旅遊認證有哪些？",
        "台灣推動多項生態旅遊認證。環保署推動的環保旅館認證與綠色餐廳認證。林業署推動的生態旅遊地認證，結合社區發展與自然資源保育。台灣生態旅遊協會推動生態旅遊標章。此外還有綠色旅行標章、田園民宿認證等。生態旅遊路線涵蓋部落體驗（司馬庫斯、鎮西堡）、賞鯨豚生態遊、森林療癒（太平山、阿里山）等。"
    ))
    records.append(qa(
        "What is Taiwan's approach to climate change policy?",
        "Taiwan passed the Climate Change Response Act (氣候變遷因應法) in 2023, setting a net-zero emissions target by 2050. Key strategies include expanding renewable energy to 60-70% of electricity generation by 2050, promoting electric vehicles, improving industrial energy efficiency, and developing carbon capture technology. Taiwan plans to establish a carbon fee system (碳費) starting in 2025, initially targeting major emitters. The '12 Key Strategies' outline pathways for green energy, circular economy, and sustainable transportation."
    ))
    records.append(qa(
        "台灣的養殖漁業有什麼特色？",
        "台灣水產養殖技術先進，主要養殖物種包括：虱目魚（台南沿海）、石斑魚（屏東，年產量約2萬噸，全球領先）、台灣鯛（改良吳郭魚，外銷美國、日本、韓國）、白蝦、鰻魚（外銷日本為大宗）、牡蠣（鮮蚵，彰化與雲林沿海）、文蛤。台灣發展室內循環水養殖系統（RAS）與智慧養殖技術，減少環境衝擊並提高產能。"
    ))

    return records
