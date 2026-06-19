"""
Massive NTNU and Taiwan Q&A dataset (300+ records).

Covers Taiwan geography, history, culture & society, and deeper NTNU topics.
Sources: NTNU official site, Wikipedia, Taiwan government publications.
"""

from typing import Dict, List
from ntnu_extended_records import qa


def get_massive_records1() -> List[Dict]:
    """Generate approximately 300+ Q&A records about Taiwan and NTNU.

    Returns:
        List of ChatML-format record dicts.
    """
    records: List[Dict] = []

    # =========================================================================
    # SECTION 1: TAIWAN GEOGRAPHY (~80 records)
    # =========================================================================

    # --- Special Municipalities ---
    records.append(qa(
        "臺灣有哪六個直轄市？",
        "臺灣現有六個直轄市：臺北市（首都，政治經濟中心）、新北市（人口最多的城市）、桃園市（國際機場所在地）、臺中市（中部核心城市）、臺南市（歷史古都）、高雄市（南部最大港都）。2010年縣市合併升格後形成目前的六都格局，合計人口約占全臺灣70%以上。"
    ))
    records.append(qa(
        "臺北市有什麼特色？",
        "臺北市為臺灣首都，面積約271.8平方公里，人口約250萬。擁有臺北101、故宮博物院、中正紀念堂等重要地標。交通便捷，臺北捷運涵蓋全市與周邊衛星城市。陽明山國家公園位於市區北側，可快速抵達。商圈多元，從信義區精品百貨到西門町流行文化應有盡有。"
    ))
    records.append(qa(
        "新北市有什麼特色？",
        "新北市是臺灣人口最多的城市，約400萬人，環繞臺北市形成「北北基桃」生活圈。地勢多樣，有淡水漁人碼頭、九份山城、野柳女王頭等知名景點。平溪天燈節、貢寮海洋音樂祭為年度盛事。2010年由臺北縣改制為直轄市。"
    ))
    records.append(qa(
        "桃園市有什麼特色？",
        "桃園市為國門之都，臺灣桃園國際機場所在地，人口約230萬。著名景點包括大溪老街、石門水庫、拉拉山神木群。桃園也是臺灣重要客家聚落，客家文化節為年度盛事。近年發展航空城計畫，推動機場經濟與產業升級。"
    ))
    records.append(qa(
        "臺中市有什麼特色？",
        "臺中市為臺灣中部核心城市，人口約280萬，以優良氣候著稱，全年日照充足，有「陽光城市」之稱。知名景點包括逢甲夜市、彩虹眷村、高美濕地、國家歌劇院。臺中是臺灣自行車製造重鎮（捷安特總部），也是糕餅產業中心（太陽餅、鳳梨酥）。氣候宜人，很少颱風直接侵襲。"
    ))
    records.append(qa(
        "臺南市有什麼特色？",
        "臺南市為臺灣歷史古都，1624年荷蘭人建立熱蘭遮城開啟歷史新頁，人口約185萬。臺灣諺語「一府二鹿三艋舺」中的「府」即指臺南府城。赤崁樓、安平古堡、孔廟為必訪古蹟。臺南以美食聞名，牛肉湯、碗粿、擔仔麵、鱔魚意麵、棺材板均發源於此。"
    ))
    records.append(qa(
        "高雄市有什麼特色？",
        "高雄市為臺灣南部最大港都，人口約270萬，擁有高雄港（臺灣最大國際商港）與高雄國際機場。愛河貫穿市區，駁二藝術特區為文創地標。85大樓、美麗島捷運站光之穹頂為知名建築。近年推動港區轉型，從重工業城市轉向觀光與文創。氣候溫暖，冬季尤其宜人。"
    ))
    records.append(qa(
        "What are the six special municipalities of Taiwan?",
        "Taiwan has six special municipalities (直轄市): Taipei City (capital), New Taipei City (most populous), Taoyuan City (home to Taiwan Taoyuan International Airport), Taichung City (central hub), Tainan City (historical capital), and Kaohsiung City (major port). They were formed through county-city mergers in 2010 and house over 70% of Taiwan's population."
    ))
    records.append(qa(
        "高雄港的歷史和規模如何？",
        "高雄港是臺灣最大國際商港，位於高雄市南端，扼臺灣海峽與巴士海峽要衝。始建於日治時期（1908年開港），1970–80年代經濟起飛時達到巔峰，貨櫃吞吐量曾排名世界前三。現為國際貨櫃樞紐港，兼營觀光（駁二特區、流行音樂中心）。港區設有高雄港務分公司管理。"
    ))

    # --- Counties and major cities ---
    records.append(qa(
        "臺灣有哪些縣？",
        "臺灣現有13個縣：基隆市（省轄市）、新竹市（省轄市）、嘉義市（省轄市）、新竹縣、苗栗縣、彰化縣、南投縣、雲林縣、嘉義縣、屏東縣、宜蘭縣、花蓮縣、臺東縣。加上六都共22個縣市行政區。離島另有澎湖縣、金門縣、連江縣。"
    ))
    records.append(qa(
        "宜蘭縣有什麼特色？",
        "宜蘭縣位於臺灣東北部，以雪山隧道與臺北連接，車程約40分鐘。以溫泉（礁溪）、冷泉（蘇澳）、童玩節（冬山河）聞名。蘭陽平原三面環山、東臨太平洋，生產高品質的蔥、茶葉（冬山素馨茶）和鴨賞（地方特產）。氣候多雨，是臺灣東北季風的第一道迎風面。"
    ))
    records.append(qa(
        "花蓮縣有什麼特色？",
        "花蓮縣位於臺灣東部，面積約4,629平方公里，為臺灣面積最大的縣。太魯閣國家公園為國際級景點，大理石峽谷景觀壯麗。花蓮也是原住民族重要聚落，阿美族、太魯閣族、布農族等族群文化豐富。七星潭、清水斷崖、六十石山金針花海為觀光亮點。"
    ))
    records.append(qa(
        "臺東縣有什麼特色？",
        "臺東縣位於臺灣東南部，以慢活步調與原民文化著稱。擁有三仙台、知本溫泉、綠島等著名景點。每年夏季舉辦熱氣球嘉年華與豐年祭，吸引大量觀光客。蘭嶼達悟族文化獨具特色。池上米、釋迦（水果）為地方特產。"
    ))
    records.append(qa(
        "屏東縣有什麼特色？",
        "屏東縣位於臺灣最南端，北回歸線以南，氣候炎熱。墾丁國家公園為臺灣首座國家公園，擁有美麗沙灘與珊瑚礁生態。東港黑鮪魚季、萬巒豬腳、恆春民謠為文化特色。屏東也是農業大縣，生產蓮霧、芒果、鳳梨等熱帶水果。大鵬灣國家風景區提供水上活動。"
    ))
    records.append(qa(
        "南投縣有什麼特色？",
        "南投縣為臺灣唯一不靠海的縣，位於中央山脈心臟地帶。日月潭是臺灣最大天然湖泊，阿里山（部分在南投）聞名中外。溪頭、杉林溪為避暑勝地。清境農場呈現歐洲風情。南投也是烏龍茶（凍頂茶）發源地。921地震震央位於南投集集，對當地影響深遠。"
    ))
    records.append(qa(
        "彰化縣有什麼特色？",
        "彰化縣位於臺灣中部，以八卦山聞名，大佛雕像為經典地標。鹿港小鎮保存完整古蹟與傳統工藝（天后宮、老街、摸乳巷）。田中馬拉松聞名全臺，帶動路跑風氣。彰化也是農業大縣，生產葡萄、花卉、扇貝等，花卉年節拍賣規模驚人。"
    ))
    records.append(qa(
        "雲林縣有什麼特色？",
        "雲林縣為臺灣農業首都，生產稻米、蔬菜、甘蔗等大宗農產品。北港朝天宮為全國媽祖信仰總廟之一，每年進香人潮湧入。古坑生產臺灣咖啡，華山地區為咖啡與夜景勝地。西螺大橋橫跨濁水溪，為歷史地標。近年推動農業觀光與地方創生。"
    ))
    records.append(qa(
        "苗栗縣有什麼特色？",
        "苗栗縣為客家文化重鎮，客家人口比例超過60%。三義木雕、大湖草莓、南庄老街、泰安溫泉為觀光亮點。雪霸國家公園位於縣境，觀霧遊憩區景色優美。客家桐花祭於每年4–5月舉行，桐花紛飛如雪。苗栗也是風力發電重要設置區域。"
    ))
    records.append(qa(
        "基隆市有什麼特色？",
        "基隆市為臺灣北部重要港口城市，以雨都著稱（年均降雨日約210天）。基隆港為國際商港，也曾是臺灣主要門戶。廟口夜市以泡泡冰、營養三明治、咖哩炒麵聞名。和平島、八斗子、望幽谷為海岸景點。基隆中元祭為年度大型傳統活動。"
    ))
    records.append(qa(
        "新竹市和新竹縣有什麼不同？",
        "新竹市為省轄市，以科技業著稱，新竹科學園區聚集半導體、光電等高科技產業，有「臺灣矽谷」之稱。新竹縣則以農業與客家文化為主，新埔柿餅、關西仙草、北埔擂茶為地方特產。兩者於1982年分治，形成一市一縣格局。竹北市為縣治所在，近年發展快速。"
    ))
    records.append(qa(
        "嘉義市和嘉義縣有什麼不同？",
        "嘉義市為省轄市，以雞肉飯聞名，市區有嘉義公園、檜意森活村（日式木造建築群）等景點。嘉義縣以農業為主，阿里山森林鐵路為世界級景點，達娜伊谷、奮起湖、太平雲梯為觀光熱點。嘉義縣治設於太保市，高鐵嘉義站位於太保。"
    ))

    # --- Mountain ranges ---
    records.append(qa(
        "臺灣有哪些主要山脈？",
        "臺灣有五大山脈，大致呈南北走向：中央山脈（縱貫全島，有「臺灣屋脊」之稱）、雪山山脈（北部，雪山主峰3,886m）、玉山山脈（中部，玉山主峰3,952m為東亞最高峰）、阿里山山脈（中南部，以森林鐵路與日出聞名）、海岸山脈（東部，花東縱谷東側）。合歡山為中央山脈與雪山山脈交會處，為知名賞雪景點。"
    ))
    records.append(qa(
        "玉山有多高？有什麼特色？",
        "玉山主峰海拔3,952公尺，為東亞最高峰（含日本富士山）。玉山國家公園涵蓋範圍包括南投、嘉義、高雄、花蓮四縣。登玉山是許多臺灣人的「必做三件事」之一，每年申請攀登人數以萬計。山頂於日治時期建有于右任銅像（現為新中橫氣象站）。阿里山脈與玉山山脈以塔塔加鞍部相接。"
    ))
    records.append(qa(
        "阿里山有什麼特色？",
        "阿里山位於臺灣中南部，隸屬阿里山山脈，以五奇著稱：日出、雲海、森林鐵路、晚霞、神木。阿里山森林鐵路為世界三大登山鐵路之一，由海拔30公尺爬升至2,216公尺。每年3–4月櫻花季吸引大量遊客。鄒族部落位於阿里山區，保留原民文化。"
    ))
    records.append(qa(
        "中央山脈有哪些著名山峰？",
        "中央山脈縱貫臺灣南北，有「臺灣脊樑」之稱，著名山峰包括：武嶺（合歡山主峰，3,417m，公路最高點）、奇萊山（以險峻著稱，有「黑色奇萊」之稱）、能高山、丹大山、秀姑巒山、關山、北大武山。能高安東軍縱走為經典長程登山路線。"
    ))
    records.append(qa(
        "合歡山為什麼是熱門景點？",
        "合歡山位於南投與花蓮交界，海拔3,000公尺以上，為臺灣最容易抵達的高山景區。冬季降雪時吸引大量遊客上山賞雪（雪季實施交通管制）。合歡群峰包括合歡主峰、東峰、北峰、西峰、石門山等，多條步道適合高山健行。武嶺（3,275m）為臺灣公路最高點。"
    ))
    records.append(qa(
        "What are Taiwan's five major mountain ranges?",
        "Taiwan has five major north-south mountain ranges: the Central Mountain Range (the backbone of Taiwan), the Xueshan Range (northern Taiwan), the Yushan Range (central, containing Jade Mountain at 3,952m), the Alishan Range (south-central, famous for its forest railway), and the Coastal Range (eastern Taiwan). They make Taiwan one of the most mountainous places in the world, with over 200 peaks above 3,000 meters."
    ))

    # --- Rivers and lakes ---
    records.append(qa(
        "臺灣最長的河流是什麼？",
        "濁水溪是臺灣最長河流，全長約186.6公里，發源於中央山脈合歡山，流經南投、雲林、彰化，由彰化大城鄉與雲林麥寮鄉之間入臺灣海峽。因河流含沙量高而得名「濁水」。中游建有日月潭水庫與明潭發電廠。濁水溪也是臺灣中部的天然分界線。"
    ))
    records.append(qa(
        "臺灣有哪些主要河流？",
        "臺灣主要河流多呈東西走向，包括：濁水溪（最長，186.6km）、高屏溪（流域最廣）、淡水河（流經臺北）、大甲溪（水力發電重鎮）、曾文溪（南部重要河川）、秀姑巒溪（泛舟勝地）、立霧溪（太魯閣峽谷）。因中央山脈南北縱貫，河流東西分流，大多坡陡流急。"
    ))
    records.append(qa(
        "日月潭有什麼特色？",
        "日月潭位於南投縣魚池鄉，為臺灣最大天然湖泊（面積約7.93平方公里），湖面海拔約748公尺。以拉魯島為界，北半部形如日輪、南半部形如月鉤而得名。日月潭萬人泳渡為年度盛事。周邊有向山遊客中心、玄光寺、伊達邵部落等景點。紅茶（日月潭紅玉/臺茶18號）為當地特產。"
    ))
    records.append(qa(
        "曾文水庫的功能是什麼？",
        "曾文水庫位於嘉義縣大埔鄉，為臺灣最大水庫（有效容量約5億立方公尺），建於1973年。主要功能為供水（嘉南地區農業灌溉與民生用水）、發電（水力發電廠）、防洪與觀光。水庫周邊的曾文水庫風景區提供遊湖、露營等活動。曾文溪全長138.5公里，為臺灣第四長河。"
    ))
    records.append(qa(
        "秀姑巒溪為什麼適合泛舟？",
        "秀姑巒溪位於花蓮縣，全長約103公里，為臺灣東部重要河川。其中瑞穗至長虹橋段約24公里，河床多礫石、水流湍急、落差適中，為臺灣最知名的泛舟路線，每年5–10月為泛舟旺季。秀姑巒溪因切穿海岸山脈形成峽谷，長虹橋為經典泛舟終點地標。"
    ))

    # --- National parks ---
    records.append(qa(
        "臺灣有哪幾個國家公園？",
        "臺灣現有9座國家公園：墾丁（1984，首座）、玉山（1985）、陽明山（1985）、太魯閣（1986）、雪霸（1992）、金門（1995）、東沙環礁（2007）、台江（2009）、澎湖南方四島（2014）。涵蓋高山、海洋、離島、濕地等多樣生態系，合計面積約占臺灣陸域8.5%。"
    ))
    records.append(qa(
        "太魯閣國家公園有什麼特色？",
        "太魯閣國家公園位於花蓮、臺中、南投三縣交界，以大理岩峽谷景觀聞名於世。立霧溪切割形成深度超過1,000公尺的峽谷，燕子口、九曲洞、長春祠為經典步道。東西橫貫公路（中橫）穿越園區，其開鑿歷史見證臺灣戰後開發史。園區保存泰雅族文化遺跡。"
    ))
    records.append(qa(
        "墾丁國家公園有什麼特色？",
        "墾丁國家公園位於屏東縣恆春半島，為臺灣首座國家公園（1984年設立）。三面臨海，擁有珊瑚礁海岸、砂灘、石灰岩洞穴等特殊地景。鵝鑾鼻燈塔為臺灣最南端地標。每年春季的「墾丁春吶」音樂祭吸引年輕族群。海域生態豐富，為潛水勝地。落山風為恆春半島特殊氣候現象。"
    ))
    records.append(qa(
        "陽明山國家公園有什麼特色？",
        "陽明山國家公園位於臺北市北側，為大屯火山群的所在地，擁有火山地形、溫泉與硫磺礦。小油坑、擎天崗、竹子湖為熱門景點，春季花季（櫻花、杜鵑、海芋）極負盛名。陽明山也是臺北市民最便利的高山踏青去處，海拔200–1,120公尺，可俯瞰臺北盆地。"
    ))
    records.append(qa(
        "What are the must-visit national parks in Taiwan?",
        "Taiwan has nine national parks. The most famous are: **Taroko Gorge** (marble canyons in Hualien), **Yushan** (East Asia's highest peak at 3,952m), **Kenting** (tropical beaches and coral reefs at the southern tip), **Yangmingshan** (volcanic hot springs near Taipei), and **Shei-Pa** (alpine ecology with the majestic Mt. Sylvia). Each offers unique landscapes accessible via well-maintained trails."
    ))

    # --- Climate and natural resources ---
    records.append(qa(
        "臺灣的氣候類型是什麼？",
        "臺灣氣候主要屬亞熱帶季風氣候，南部屬熱帶季風氣候。北回歸線（23.5°N）通過嘉義縣與花蓮縣之間。北部全年有雨（冬季東北季風），南部冬乾夏雨（以梅雨和颱風為主）。山區氣溫隨海拔遞減，玉山頂年均溫約3.5°C。颱風每年平均侵臺3–4次，主要集中于7–9月。"
    ))
    records.append(qa(
        "臺灣有哪些自然資源？",
        "臺灣自然資源相對有限，較重要的包括：森林資源（覆蓋率約60%）、水力資源（河川坡陡流急，水力發電佔比約3–4%）、地熱資源（大屯火山區、清水地熱）、礦產（少量煤、大理石、石灰石、金、銅等，大都已停採）。漁業資源豐富，為遠洋漁業大國。農產品以稻米、茶葉、水果為大宗。"
    ))
    records.append(qa(
        "臺灣的生態有什麼特色？",
        "臺灣因地形多變，從海岸到高山涵蓋多種生態系：海岸林、溼地、平原農田、低海拔闊葉林、中海拔混合林、高海拔針葉林與高山苔原。特有種比例極高，如臺灣黑熊、臺灣藍鵲、臺灣山椒魚、櫻花鉤吻鮭（國寶魚，棲息於雪霸國家公園七家灣溪）。蝴蝶種類密度居世界之冠。"
    ))
    records.append(qa(
        "臺灣為什麼地震頻繁？",
        "臺灣位於環太平洋火山地震帶，因菲律賓海板塊與歐亞板塊碰撞而形成。板塊交界處位於花東縱谷，每年約有數千次有感或無感地震。921大地震（1999年，芮氏規模7.3）造成嚴重災情，此後大幅強化建築耐震法規與防災教育。地震研究由中央氣象局與中研院地球科學研究所負責。"
    ))

    # --- Islands ---
    records.append(qa(
        "金門有什麼特色？",
        "金門位於臺灣海峽西側，靠近中國大陸廈門，由金門本島、烈嶼等島嶼組成。歷經古寧頭戰役、八二三砲戰等戰役，保留大量戰地遺跡（坑道、碉堡、軌條砦）。金門酒廠生產高粱酒聞名全球。閩南建築（得月樓、山后民俗文化村）為文化資產。金門國家公園涵蓋戰役紀念地與自然生態。"
    ))
    records.append(qa(
        "馬祖有什麼特色？",
        "馬祖（連江縣）位於臺灣海峽北端，由南竿、北竿、東莒、西莒、東引等島嶼組成。以戰地文化、藍眼淚（夜光藻生物發光現象）與閩東建築（石頭老屋）聞名。馬祖酒廠生產馬祖高粱與老酒，淡菜、魚麵為地方美食。每年4–6月為賞藍眼淚季節。北海坑道等軍事遺跡可划獨木舟。"
    ))
    records.append(qa(
        "澎湖有什麼特色？",
        "澎湖位於臺灣海峽中，由90座島嶼組成，以玄武岩地質景觀（柱狀節理）與咾咕石建築聞名。澎湖跨海大橋連接白沙與西嶼。花火節為年度觀光盛事。桶盤嶼、七美雙心石滬、吉貝沙尾為經典景點。澎湖海洋資源豐富，以海鮮（牡蠣、小管、海膽）為飲食特色。冬季東北季風強勁。"
    ))
    records.append(qa(
        "綠島有什麼特色？",
        "綠島位於臺東縣東方太平洋，以潛水與溫泉聞名。朝日溫泉為世界少數海底溫泉之一。綠島過去為政治犯監獄所在地（白色恐怖時期），現轉型為觀光島，人權紀念園區見證歷史。柴口、石朗為優質潛水點，珊瑚生態豐富。島上可騎機車環島，約20公里，一小時內可環繞一圈。"
    ))
    records.append(qa(
        "蘭嶼有什麼特色？",
        "蘭嶼位於臺東縣東南太平洋，為達悟族（雅美族）原住民族世居地，保有完整的傳統文化（拼板舟、飛魚祭、地下屋）。蘭嶼核廢料儲存場為重大爭議議題。島上無紅綠燈，生活步調原始悠閒。東清灣日出、青青草原夕陽、野銀冷泉為觀光景點。飛魚季於每年2–6月舉行，為達悟族最重要祭祀文化。"
    ))
    records.append(qa(
        "What are Taiwan's main outlying islands?",
        "Taiwan's main outlying islands are: **Kinmen** (near Xiamen, known for wartime history and kaoliang liquor), **Matsu** (blue tears bioluminescence and Eastern Fujian architecture), **Penghu** (basalt columns and the annual fireworks festival), **Green Island** (diving and hot springs), and **Orchid Island** (indigenous Tao culture and flying fish festivals). Each has distinct geography and culture."
    ))

    # --- East vs West Coast ---
    records.append(qa(
        "臺灣東海岸和西海岸有什麼不同？",
        "臺灣東海岸以斷層海岸為主，緊鄰太平洋，陡峭深邃，景觀壯麗（清水斷崖、太魯閣、三仙台）。西海岸多沙灘、潟湖、濕地，地勢平緩，潮間帶寬廣（高美濕地、七股潟湖）。東岸因板塊碰撞持續抬升，西岸則有廣大沖積平原（嘉南平原、彰化平原）。東岸以觀光與漁業為主，西岸為農業與工業重鎮。"
    ))
    records.append(qa(
        "花東縱谷是什麼？",
        "花東縱谷位於中央山脈與海岸山脈之間，北起花蓮市、南至臺東市，全長約180公里。為菲律賓海板塊與歐亞板塊交界帶，地形上為狹長谷地。縱谷是臺灣主要農業區之一，生產稻米（池上米、關山米）、水果（鳳梨、釋迦）與茶葉。鹿野高臺為熱氣球活動據點。原住民族（阿美族、布農族）在此聚居。"
    ))

    # --- City characteristics ---
    records.append(qa(
        "為什麼臺中有「陽光城市」之稱？",
        "臺中全年日照時數為臺灣主要城市之冠，平均約2,100小時。因位於中央山脈背風側，颱風直接侵襲機率較低。氣候溫和，年均溫約23°C。宜居條件吸引人口持續成長，已成為臺灣第二大都市（人口約280萬）。七期重劃區為新興商業中心，國家歌劇院為重要文化地標。"
    ))
    records.append(qa(
        "臺灣有哪些夜市是必去的？",
        "臺灣必去夜市包括：臺北士林夜市（規模最大，蚵仔煎、大腸包小腸）、臺中逢甲夜市（創新小吃發源地，明倫蛋餅、大腸包小腸）、臺南花園夜市（以占地廣、選擇多著稱）、高雄六合夜市（海鮮與木瓜牛奶）、高雄瑞豐夜市（年輕人最愛）。每個夜市各有特色，但共通點為物美價廉、充滿臺灣活力。"
    ))

    # =========================================================================
    # SECTION 2: TAIWAN HISTORY (~80 records)
    # =========================================================================

    # --- Prehistoric ---
    records.append(qa(
        "臺灣的史前時期有哪些重要文化？",
        "臺灣史前文化可追溯至舊石器時代晚期的長濱文化（約5萬年前–5千年前），以臺東長濱八仙洞遺址為代表。新石器時代有大坌坑文化（約7千–5千年前，臺灣最早農業文化）、圓山文化（臺北，以貝塚聞名）、卑南文化（臺東，石板棺與玉器工藝，約3千年前）。十三行遺址（新北八里）為臺灣最早鐵器文化（約1,800–500年前）。"
    ))
    records.append(qa(
        "南島語系與臺灣的關係是什麼？",
        "語言學研究證實臺灣為南島語系的起源地之一。臺灣原住民族語言保存古南島語特徵，國際學者如白樂思（Robert Blust）等認為原南島語（Proto-Austronesian）約6千年前起源於臺灣。現有16族原住民的語言分屬泰雅語群、排灣語群等，是南島語系最古老且多元的分支。"
    ))

    # --- Dutch and Spanish colonial period ---
    records.append(qa(
        "荷蘭人在臺灣的統治是怎樣的？（1624–1662）",
        "荷蘭東印度公司（VOC）於1624年在臺南建立熱蘭遮城（Zeelandia），開啟臺灣歷史上的荷蘭統治時期。荷蘭人招募漢人來臺開墾，以蔗糖與鹿皮為主要貿易商品，並引進牛隻與農業技術。傳教士編寫西拉雅語字典，以羅馬拼音記錄原住民語言（新港文書）。1662年鄭成功圍攻熱蘭遮城，結束荷蘭統治。"
    ))
    records.append(qa(
        "西班牙人在臺灣的歷史是怎樣的？",
        "西班牙人於1626年佔領臺灣北部（今基隆、淡水），建立聖薩爾瓦多城（基隆和平島）與聖多明哥城（淡水，今紅毛城前身）。西班牙人主要目的是確保馬尼拉與日本之間的貿易航線，並對原住民進行傳教。1642年荷蘭人北上驅逐西班牙勢力，結束其16年的北部統治。"
    ))
    records.append(qa(
        "What is Fort Zeelandia and why is it important?",
        "Fort Zeelandia (熱蘭遮城) was built by the Dutch East India Company in 1624 on the coast of modern-day Tainan. It served as the administrative center of Dutch rule in Taiwan (1624–1662) and was a key trading post between China, Japan, and Southeast Asia. Koxinga (Zheng Chenggong) besieged the fort in 1661 and captured it in 1662, ending Dutch colonial rule. Today, the remains are a major historical site known as Anping Fort (安平古堡)."
    ))

    # --- Koxinga / Ming-Zheng period ---
    records.append(qa(
        "鄭成功在臺灣的歷史地位如何？",
        "鄭成功（國姓爺）於1661年率軍自福建渡海，圍攻荷蘭人據守的熱蘭遮城，1662年成功收復臺灣。他以臺灣為反清復明根據地，建立明鄭政權，實施屯田政策、推行漢人教育、建立孔廟。鄭氏時期引入大量漢人移民與中國行政制度，為臺灣漢人社會奠定基礎。鄭成功逝世後由其子鄭經、孫鄭克塽繼位，1683年降清。"
    ))
    records.append(qa(
        "明鄭時期臺灣的建設有哪些？",
        "明鄭時期（1662–1683）鄭成功與其子鄭經在臺灣的建設包括：實施軍屯與民屯制度，解決糧食問題；設立承天府（今臺南）為行政中心；建立臺灣第一座孔廟（「全臺首學」）推行儒學教育；獎勵蔗糖與鹿皮貿易。鄭經時期進一步擴展至北部（淡水、基隆）與東部，並與英國東印度公司簽約貿易。"
    ))

    # --- Qing dynasty rule ---
    records.append(qa(
        "清朝統治臺灣的時期有哪些重要發展？",
        "清朝於1683年施琅攻佔臺灣後納入版圖，初期採消極隔離政策，嚴禁漢人攜眷來臺。19世紀後因列強覬覦，轉為積極治理：1885年臺灣建省，劉銘傳任首任巡撫，推動鐵路（基隆–新竹）、電報、郵政、煤礦等現代化建設。甲午戰爭（1894–1895）清廷戰敗，簽訂馬關條約將臺灣割讓日本，結束212年清朝統治。"
    ))
    records.append(qa(
        "臺灣為什麼會有「一府二鹿三艋舺」的說法？",
        "「一府二鹿三艋舺」為清代臺灣三大港市的繁華排名。「一府」指臺南府城（今臺南市），為全臺政治經濟中心；「二鹿」指鹿港（今彰化），為中部重要貿易口岸；「三艋舺」指淡水河畔的艋舺（今臺北萬華），為北部商業樞紐。三者在18–19世紀主導臺灣對大陸貿易，也反映了臺灣開發由南向北的歷史軌跡。"
    ))
    records.append(qa(
        "劉銘傳對臺灣現代化有什麼貢獻？",
        "劉銘傳於1885年出任臺灣首任巡撫，推動多項現代化建設：興建臺灣第一條鐵路（基隆–臺北–新竹）、架設電報線、設立郵政總局、開採煤礦、設立西學堂與電報學堂。基隆港與高雄港的近代化工程亦由他奠基。劉銘傳被譽為「臺灣現代化之父」，其改革成果為日治時期進一步發展打下基礎。"
    ))

    # --- Japanese colonial period ---
    records.append(qa(
        "日治時期臺灣的教育發展如何？",
        "日治時期（1895–1945）臺灣教育體系逐步建立：初設國語傳習所（日語），後發展為公學校（漢人）與蕃人教育所（原住民）。1922年《臺灣教育令》確立中等以上教育體系，成立臺北高等學校（師大前身）、臺北帝國大學（臺大前身）、臺北師範學校等。但教育機會不均等，臺灣人受高等教育比例遠低於日本人。日語普及率至終戰時約70%。"
    ))
    records.append(qa(
        "日治時期臺灣的文化發展有哪些？",
        "日治時期臺灣文化蓬勃發展，尤其1920年代以後：蔣渭水等人成立臺灣文化協會（1921年），推動白話文運動與民族意識啟蒙。文學方面有賴和、楊逵、張我軍等作家。建築上引進西洋與現代主義風格，如總統府、監察院、臺中車站等。美術方面，陳澄波、廖繼春等畫家入選日本帝展。臺語流行歌在此時期萌芽（望春風、雨夜花）。"
    ))
    records.append(qa(
        "日治時期臺灣的基礎建設有哪些？",
        "日治時期臺灣基礎建設成果顯著：縱貫鐵路（基隆–高雄）1908年全線通車；基隆港、高雄港、花蓮港等港口現代化；嘉南大圳（烏山頭水庫，八田與一設計）灌溉嘉南平原15萬公頃農田；日月潭水力發電所（1934年）供應全臺工業用電。都市計劃方面，臺北、臺中、高雄進行棋盤式道路規劃，自來水、電力普及率大幅提升。"
    ))
    records.append(qa(
        "八田與一對臺灣有什麼貢獻？",
        "八田與一為日治時期日本水利工程師，1920–1930年間規劃並監造烏山頭水庫與嘉南大圳水利系統，使嘉南平原15萬公頃看天田變成旱澇保收的良田。他對臺灣農業貢獻極大，被譽為「嘉南大圳之父」。其妻外代樹於戰後投水殉情。烏山頭水庫現設有八田與一紀念園區，每年5月8日（他的忌日）舉行追思會。"
    ))
    records.append(qa(
        "What was the Taipei Higher School (Taihoku Higher School)?",
        "Taihoku Higher School (臺北高等學校) was founded in 1922 during Japanese rule as a preparatory institution for higher education. It followed the Japanese higher school system modeled on the German gymnasium, offering a liberal arts education. The school's main building is now the administrative building of National Taiwan Normal University. After WWII, it was reorganized into Taiwan Provincial Teachers College, eventually becoming NTNU. The school's legacy remains a key part of NTNU's institutional identity."
    ))

    # --- Post-war reconstruction ---
    records.append(qa(
        "臺灣戰後初期的情況如何？（1945–1960s）",
        "1945年日本投降，國民政府接收臺灣，設臺灣省行政長官公署（陳儀為長官）。初期面臨通貨膨脹、失業、物資短缺等問題。1947年發生二二八事件，本省人與外省人矛盾激化。1949年國民政府遷臺，實施戒嚴，展開白色恐怖。1950–1960年代推動土地改革（三七五減租、公地放領、耕者有其田），以農業培養工業，奠定經濟起飛基礎。美援在此時期扮演重要角色。"
    ))
    records.append(qa(
        "二二八事件發生的原因和影響是什麼？",
        "二二八事件發生於1947年2月27日–3月初，導火線為2月27日臺北緝煙員打死民眾，2月28日群眾抗議遭鎮壓。事件迅速蔓延全臺，本省人攻擊外省人，國民政府調兵鎮壓造成大規模傷亡（估計死傷1萬至3萬人）。事件造成臺灣社會長期的省籍裂痕與政治創傷。1995年政府設立二二八紀念碑與紀念館，每年2月28日為和平紀念日，推動轉型正義與歷史還原。"
    ))
    records.append(qa(
        "白色恐怖時期是什麼？",
        "白色恐怖時期指1949–1987年臺灣戒嚴期間，政府以「懲治叛亂條例」與「動員戡亂時期臨時條款」為法律基礎，大規模鎮壓政治異議份子。包括槍決、長期監禁、思想改造等。綠島曾設政治犯監獄（綠洲山莊）。美麗島事件（1979年）為最具影響力的民主運動事件，林義雄、施明德、陳水扁、呂秀蓮等參與者後成為民進黨核心。1990年代後逐步平反與賠償。"
    ))
    records.append(qa(
        "戒嚴時期對臺灣社會有什麼影響？",
        "臺灣戒嚴時期自1949年5月20日起至1987年7月15日止，長達38年又56天，為世界最長戒嚴之一。期間限制人民集會結社自由、實施報禁與黨禁、禁組新政黨。威權統治下經濟快速發展，但政治自由受到嚴重限制。1987年蔣經國解除戒嚴，隨後陸續開放黨禁、報禁，開啟民主化進程。"
    ))

    # --- Economic miracle ---
    records.append(qa(
        "臺灣經濟奇蹟是怎麼發生的？（1960s–1990s）",
        "臺灣經濟奇蹟指1960–1990年代臺灣從農業社會快速轉型為已開發經濟體的過程。1950–60年代以進口替代與出口導向政策為核心，設立加工出口區（高雄、楠梓、臺中）。1970年代推動十大建設，包括中山高速公路、中正國際機場、鐵路電氣化、煉鋼廠、造船廠等重工業。1980年代設立新竹科學園區，轉向半導體與高科技產業。1990年代臺灣成為全球資訊科技硬體製造中心。"
    ))
    records.append(qa(
        "臺灣的十大建設有哪些？",
        "十大建設由行政院長蔣經國於1974年推動，包含：中山高速公路、縱貫鐵路電氣化、北迴鐵路、桃園國際機場、臺中港、蘇澳港、高雄造船廠（中國造船）、高雄煉鋼廠（中鋼）、石油化學工業（中油）、核能發電廠（核一）。總經費約新臺幣2,400億元，奠定了臺灣1970年代以後的工業化與現代化基礎。"
    ))
    records.append(qa(
        "新竹科學園區對臺灣經濟的影響？",
        "新竹科學園區成立於1980年，為臺灣高科技產業搖籃。園區內聚集臺積電、聯電、聯發科、鴻海等半導體與資訊大廠。2020年代全球半導體供應鏈中臺灣佔據關鍵地位，臺積電為全球最大晶圓代工廠。竹科的成功模式後來複製到臺中（中部科學園區）與臺南（南部科學園區），形成臺灣西部科技走廊。"
    ))
    records.append(qa(
        "How did Taiwan transform from agriculture to high-tech industry?",
        "Taiwan's transformation happened in stages: (1) 1950s–60s: Land reform and import-substitution industrialization, (2) 1960s–70s: Export processing zones and labor-intensive manufacturing, (3) 1970s: Ten Major Infrastructure Projects including highways, railways, ports, and steel/petrochemical plants, (4) 1980s onward: Hsinchu Science Park and the rise of the semiconductor industry. By the 1990s, Taiwan was a global leader in IT hardware manufacturing, and by 2020 it dominated advanced semiconductor fabrication through TSMC."
    ))

    # --- Democratization ---
    records.append(qa(
        "臺灣民主化過程有哪些關鍵事件？",
        "臺灣民主化關鍵事件包括：1986年民主進步黨（民進黨）突破黨禁創黨；1987年解除戒嚴；1988年蔣經國逝世、李登輝繼任總統，推動本土化與民主改革；1991年廢除動員戡亂時期臨時條款；1992年國會全面改選；1996年舉辦首次總統直選（李登輝當選）；2000年政黨輪替，民進黨陳水扁當選總統。"
    ))
    records.append(qa(
        "臺灣第一次總統直選是什麼時候？",
        "臺灣第一次總統直選於1996年3月23日舉行，李登輝（國民黨）與連戰搭檔以54%得票率當選，擊敗民進黨彭明敏與謝長廷（21%）、林洋港與郝柏村（15%）、陳履安與王清峰（10%）。此次選舉標誌著臺灣從威權體制轉向民主政治，也是兩千年來華人社會首次國家元首直接民選。"
    ))
    records.append(qa(
        "美麗島事件的歷史意義是什麼？",
        "美麗島事件發生於1979年12月10日（世界人權日），美麗島雜誌社在高雄舉辦遊行後遭軍警鎮壓，大規模逮捕黨外運動人士。施明德、黃信介、姚嘉文、林義雄、陳菊、張俊宏、呂秀蓮等人受軍法審判。雖然多名領袖入獄，但事件激發臺灣人民對民主的追求，1980年代後民主運動急速發展，可視為臺灣民主化轉捩點。"
    ))
    records.append(qa(
        "李登輝對臺灣民主化的貢獻是什麼？",
        "李登輝於1988年繼任總統，至2000年卸任，主導臺灣民主化與本土化進程。重要貢獻包括：終止動員戡亂時期、推動國會全面改選、實施總統直選、推動本土化教育（認識臺灣課程）、倡導南向政策。李登輝被譽為「臺灣民主化推手」，但也因兩岸政策引發爭議。"
    ))

    # --- Cross-strait relations ---
    records.append(qa(
        "兩岸關係的歷史發展是怎樣的？",
        "兩岸關係自1949年國民政府遷臺後長期處於軍事對峙狀態。1987年開放探親後民間交流漸增。1991年兩岸分別成立海基會與海協會進行事務性協商。1992年達成「九二共識」各自表述。2008–2016年馬英九執政時期交流最密切，簽署ECFA等23項協議。2016年蔡英文執政後兩岸官方交流中斷。2024年賴清德當選總統，延續「務實臺獨」路線。"
    ))
    records.append(qa(
        "九二共識是什麼？",
        "九二共識（1992 Consensus）是指1992年兩岸海基會與海協會在香港會談後形成的政治默契，雙方同意以口頭方式各自表述「一個中國」原則。國民黨與中國大陸的解讀為「一個中國各自表述」；民進黨認為該共識並無共識實質，不承認其存在。此議題為兩岸關係的核心爭議之一。"
    ))
    records.append(qa(
        "臺灣現在的國際地位如何？",
        "臺灣目前未獲聯合國正式承認，但與13個國家（2025年數據）有正式外交關係，包括巴拉圭、帛琉、馬紹爾群島等。美國、日本、歐盟等主要國家與臺灣維持非官方實質關係，透過《臺灣關係法》（美）等國內法維持交流。臺灣以「中華臺北」名義參與奧運、APEC、WTO等國際組織。臺灣護照享有多國免簽待遇。"
    ))

    # --- Key events in more detail ---
    records.append(qa(
        "1947年的臺灣二二八事件經過？",
        "1947年2月27日，專賣局緝私員在臺北打傷及打死民眾，引發群眾不滿。2月28日上午群眾前往行政長官公署請願，遭衛兵開槍射擊，事件擴大成全省性衝突。全省各地民眾組成處理委員會，要求改革行政。3月初國民政府派遣軍隊來臺鎮壓，展開長達數月的清鄉。事件造成嚴重傷亡與省籍對立，成為臺灣近代史最重要創傷之一。"
    ))
    records.append(qa(
        "臺灣的核電歷史和爭議？",
        "臺灣曾運轉三座核電廠：核一（1978–2019，已除役）、核二（1982–2023，已除役）、核三（1985–2025，除役中）。1985年因民眾抗爭終止核四興建計畫。2011年福島核災後反核聲浪高漲，政府推動2025非核家園目標。擁核方主張核電為低碳電力來源；反核方強調核廢料處理難題與地震風險。"
    ))
    records.append(qa(
        "What was the February 28 Incident of 1947?",
        "The February 28 Incident (二二八事件) began on February 27, 1947, when a cigarette tax enforcement officer assaulted a woman street vendor in Taipei, sparking public outrage. The next day, protesters marched to the Governor-General's office, where soldiers opened fire, killing several. The unrest spread island-wide. The Nationalist government sent troops who conducted a brutal crackdown, killing thousands. The incident created lasting ethnic tensions between mainlanders and native Taiwanese. It was taboo for decades until the 1990s when official apologies and memorials were established."
    ))

    # =========================================================================
    # SECTION 3: TAIWAN CULTURE & SOCIETY (~70 records)
    # =========================================================================

    # --- Festivals ---
    records.append(qa(
        "臺灣的農曆新年有哪些傳統習俗？",
        "臺灣農曆新年（春節）是全年最重要的節日。習俗包括：大掃除（除舊布新）、貼春聯、圍爐吃年夜飯（發紅包壓歲錢）初一走春（拜訪親友）、初二回娘家、初四接神、初五開工。元宵節（正月十五）吃湯圓、賞花燈，臺灣燈會與平溪天燈節為年度亮點。過年期間各廟宇擠滿求福民眾。"
    ))
    records.append(qa(
        "端午節在臺灣有哪些慶祝方式？",
        "端午節（農曆五月初五）在臺灣最重要的習俗為划龍舟與吃粽子。各地舉辦龍舟賽，以臺北碧潭、高雄愛河、宜蘭冬山河等最著名。家家戶戶吃粽子（南部粽水煮口感軟黏、北部粽蒸製口感較硬）。掛艾草、喝雄黃酒、立蛋（中午時分）也是傳統習俗。端午節為國定假日。"
    ))
    records.append(qa(
        "中秋節臺灣人都怎麼過？",
        "中秋節在臺灣是僅次於春節的重要節日，傳統習俗為吃月餅、柚子、烤肉賞月。臺灣中秋烤肉文化獨樹一幟，1980年代起流行至今，親友在戶外或陽臺烤肉賞月。蛋黃酥、綠豆椪、鳳梨酥為熱門伴手禮。各地舉辦賞月活動。中秋節為國定假日。"
    ))
    records.append(qa(
        "臺灣燈會是什麼活動？",
        "臺灣燈會由交通部觀光署主辦，每年元宵節期間在臺灣不同城市舉行，為期約兩週。燈會展示大型主題花燈、競賽花燈與互動裝置藝術，主燈（當年生肖主題）開燈儀式為高潮。2023年在臺北舉辦，2024年在臺南。平溪天燈節（新北）、鹽水蜂炮（臺南）亦為元宵節重要傳統活動。"
    ))
    records.append(qa(
        "What are the three major traditional festivals in Taiwan?",
        "The three major traditional Chinese festivals celebrated in Taiwan are: **Lunar New Year** (Spring Festival, late Jan/early Feb), **Dragon Boat Festival** (5th day of 5th lunar month), and **Mid-Autumn Festival** (15th day of 8th lunar month). All three are public holidays. Each has distinctive foods and customs: New Year feasts and red envelopes, zongzi and dragon boat races at Dragon Boat, and mooncakes with barbecue on Mid-Autumn."
    ))

    # --- Folk religion ---
    records.append(qa(
        "臺灣民間信仰中最受歡迎的神明有哪些？",
        "臺灣民間信仰中媽祖（天上聖母）為最受歡迎的神明，信徒約占全臺人口三分之二。其他重要神明包括：關公（關聖帝君，忠義象徵）、土地公（地基主，最親民的神明）、觀世音菩薩（佛教兼民間信仰）、保生大帝（醫藥之神）、王爺（千歲，驅瘟之神）、玄天上帝、文昌帝君（考試之神，學生最愛祭拜）。"
    ))
    records.append(qa(
        "媽祖信仰在臺灣有多重要？",
        "媽祖（天上聖母）為臺灣信徒最多的民間信仰，全臺有超過1,000座媽祖廟。最重要的三大媽祖廟為：大甲鎮瀾宮（每年大甲媽祖遶境為全臺最大宗教活動）、北港朝天宮（臺灣媽祖信仰總廟）、鹿港天后宮（歷史最悠久）。大甲媽祖遶境進香為期9天8夜，徒步300公里，吸引百萬人次參與，被Discovery頻道列為世界三大宗教盛事之一。"
    ))
    records.append(qa(
        "臺灣的廟宇文化有什麼特色？",
        "臺灣廟宇密度極高，平均每個村里有數座廟宇。廟宇不僅是宗教場所，也是社區活動中心、老人聚會所與教育空間。建築特色包括燕尾脊、剪黏（陶瓷貼片工藝）、龍柱、交趾陶（傳統手工藝）。廟會（迎神賽會）包含陣頭（八家將、官將首）、歌仔戲、布袋戲等傳統表演，鑼鼓喧天、熱鬧非凡。"
    ))
    records.append(qa(
        "What is the Dajia Mazu Pilgrimage?",
        "The Dajia Mazu Pilgrimage (大甲媽祖遶境) is Taiwan's largest religious event, held annually in March or April. Pilgrims walk with the Mazu statue from Dajia Jenn Lann Temple in Taichung to Fengtian Temple in Xingang, Chiayi, and back — a 300km round trip over 9 days and 8 nights. Over a million people participate. The procession features traditional performances, free food for pilgrims, and is recognized as one of the world's three major religious events by Discovery Channel."
    ))

    # --- Night market culture ---
    records.append(qa(
        "臺灣的夜市文化為什麼這麼發達？",
        "臺灣夜市文化發達源於多種因素：氣候溫暖適合戶外活動、華人飲食文化重視現做現吃、都市化帶動夜間經濟、觀光產業推廣。根據統計，全臺灣有超過300個大小夜市，其中具觀光規模的約70個。夜市集美食、購物、休閒於一體，是臺灣人日常生活的一部分，也是外國遊客最愛的體驗。"
    ))
    records.append(qa(
        "臺灣有哪些知名夜市小吃？",
        "臺灣夜市經典小吃包括：蚵仔煎（牡蠣煎蛋配地瓜粉）、臭豆腐（發酵豆腐油炸，搭配泡菜）、大腸包小腸（糯米腸夾香腸）、雞排（炸雞排，比臉還大）、珍珠奶茶（發源於臺中，紅茶加牛奶與粉圓）、鹽酥雞（油炸小塊雞肉搭配九層塔）、滷肉飯、牛肉麵、刈包（虎咬豬）、芋圓（甜湯配料）。"
    ))

    # --- Tea culture ---
    records.append(qa(
        "臺灣茶的歷史和種類有哪些？",
        "臺灣茶文化始於19世紀，英國商人杜德（John Dodd）將安溪烏龍茶苗引入臺灣北部，外銷美國與歐洲。主要茶種包括：凍頂烏龍茶（南投鹿谷，發酵約30%，最具代表性）、高山茶（阿里山、梨山、杉林溪，海拔1,000m以上，清香甘甜）、文山包種茶（新北坪林，發酵約10–15%，清香）、東方美人茶（白毫烏龍，受小綠葉蟬吸食之茶菁製成，蜜香濃郁）、鐵觀音（木柵，重烘焙，濃郁炭焙味）、紅玉紅茶（日月潭，臺茶18號，薄荷肉桂香）。"
    ))
    records.append(qa(
        "臺灣高山茶為什麼特別有名？",
        "臺灣高山茶指海拔1,000公尺以上茶園生產的烏龍茶。高山氣候涼爽、雲霧繚繞、日夜溫差大，使茶葉生長緩慢、葉片肥厚、苦澀成分（兒茶素）較低、甘甜成分（茶胺酸）較高。阿里山（1,200–1,600m）、梨山（1,800–2,300m）、杉林溪（1,200–1,600m）為三大著名高山茶產區。高山茶沖泡後香氣清雅、滋味甘醇、耐泡度高。"
    ))
    records.append(qa(
        "臺灣的茶藝文化是怎樣的？",
        "臺灣茶藝（泡茶）是一種生活美學，強調「茶、水、器、火、人」五要素的和諧。常見茶具包括紫砂壺或蓋碗、聞香杯、品茗杯、茶海（公道杯）、茶則。烏龍茶沖泡流程為：溫壺→置茶→注水（約95°C）→倒掉第一泡（洗茶）→聞香→品茗。臺灣茶館（如臺北紫藤廬）結合茶藝與文化沙龍。茶道也融入現代生活，社區茶會與茶藝課程相當普遍。"
    ))

    # --- Traditional arts ---
    records.append(qa(
        "臺灣的傳統戲曲有哪些？",
        "臺灣傳統戲曲以歌仔戲為代表，為唯一發源於臺灣的劇種，使用臺語演唱，曲調優美動人，豐富的肢體語言。明華園為最知名的歌仔戲團。布袋戲（掌中戲）以黃俊雄的雲州大儒俠（史豔文）風靡全臺。京劇（國劇）亦傳入臺灣，國光劇團為主要演出團體。客家族群的採茶戲與原住民的古謠吟唱也是重要傳統表演藝術。"
    ))
    records.append(qa(
        "臺灣有哪些重要的傳統工藝？",
        "臺灣重要傳統工藝包括：三義木雕（苗栗）、鶯歌陶瓷（新北陶瓷之都）、苑裡藺草編織（苗栗）、交趾陶（廟宇裝飾工藝）、剪黏（破碎瓷片拼貼）、漆藝（蓬萊塗）、藍染（大菁植物染布）、油紙傘（美濃客家文化象徵）、刺繡（傳統閩繡與原民十字繡）、金工（金銀飾品製作）。"
    ))

    # --- Indigenous cultures ---
    records.append(qa(
        "臺灣有哪16族原住民族？",
        "臺灣目前官方認定16族原住民族，依人口多寡排列為：阿美族（Amis，最大族，約20萬人）、排灣族（Paiwan）、泰雅族（Atayal）、布農族（Bunun）、卑南族（Puyuma）、魯凱族（Rukai）、鄒族（Tsou）、賽夏族（Saisiyat）、達悟族（Yami/Tao，蘭嶼）、邵族（Thao，日月潭）、噶瑪蘭族（Kavalan）、太魯閣族（Truku）、撒奇萊雅族（Sakizaya）、賽德克族（Seediq）、拉阿魯哇族（Hla'alua）、卡那卡那富族（Kanakanavu）。原住民族總人口約58萬（占全臺人口2.5%）。"
    ))
    records.append(qa(
        "臺灣原住民族有什麼重要祭典？",
        "各原住民族有豐富祭典：阿美族豐年祭（7–9月，各部落陸續舉行，以年齡階級為核心）、布農族打耳祭（Malahtangia，4–5月，勇士狩獵與射耳儀式）、排灣族五年祭（Maleveq，每五年一次，刺球儀式）、賽夏族矮靈祭（巴斯達隘，每兩年一次，向矮靈祈求平安）、達悟族飛魚祭（2–6月，漁撈文化核心祭典）、鄒族戰祭（Mayasvi，整修會所與勇士成年禮）。"
    ))
    records.append(qa(
        "What are the indigenous peoples of Taiwan?",
        "Taiwan officially recognizes 16 indigenous groups, with a total population of about 580,000 (2.5% of Taiwan's population). The largest groups are the Amis (around 200,000), Atayal, Paiwan, and Bunun. They belong to the Austronesian language family. Indigenous cultures are particularly strong in eastern Taiwan (Hualien and Taitung) and the mountains of central and southern Taiwan. Traditional festivals like the Amis Harvest Festival and the Yami Flying Fish Festival are major cultural events."
    ))

    # --- Hakka culture ---
    records.append(qa(
        "臺灣的客家文化有什麼特色？",
        "臺灣客家人約占總人口18–20%（約400萬人），主要聚居於桃園、新竹、苗栗、高雄美濃、屏東六堆。客家話（四縣、海陸、大埔等腔調）為重要語言資產。客家文化特色包括：客家菜（鹹、香、油—薑絲炒大腸、客家小炒、梅干扣肉、粄條）、客家山歌與八音（傳統音樂）、桐花祭（每年4–5月）、敬字亭（惜字文化）。桃竹苗為客家文化核心區。"
    ))
    records.append(qa(
        "客家桐花祭是什麼活動？",
        "客家桐花祭由客家委員會主辦，每年4–5月桐花盛開季節在全臺客庄（尤其是桃竹苗）舉行。活動內容包括賞桐步道健行、客家音樂表演、手作市集、客庄小旅行。桐花象徵客家精神：團結、潔白、堅韌。桐花祭結合生態旅遊與客家文化推廣，是臺灣最具代表性的族群文化活動之一。"
    ))
    records.append(qa(
        "臺灣的語言政策與本土語言保存現狀如何？",
        "國民政府來臺初期推行國語（北京話）政策，禁止在學校使用臺語、客語與原住民族語言，導致本土語言流失。2000年後逐步推動鄉土語言教育，2018年《國家語言發展法》通過，規定各本土語言為「國家語言」。目前國小必修本土語言（臺語、客語、原住民族語或手語擇一）。臺語（Hō-ló話）使用人口最多，客語與原住民族語面臨傳承危機。"
    ))

    # --- Food culture ---
    records.append(qa(
        "臺灣的牛肉麵有什麼故事？",
        "臺灣牛肉麵為最具代表性的臺灣美食之一，融合川味、臺味與眷村（軍眷村落）飲食文化。紅燒牛肉麵以豆瓣醬、醬油、八角等香料熬製湯頭，搭配牛腱肉、酸菜與蔥花。清燉牛肉麵以牛骨慢熬，湯頭清澈鮮甜。臺北牛肉麵節每年評選最佳店家。知名店家包括林東芳、永康、廖家、老山東等。"
    ))
    records.append(qa(
        "珍珠奶茶的發明故事是什麼？",
        "珍珠奶茶（波霸奶茶）發源於1980年代的臺中。春水堂與翰林茶館都主張為發明者。春水堂創辦人劉漢介於1987年將粉圓加入冰奶茶中無意間創造出珍珠奶茶。翰林茶館則說創辦人涂宗和在1986年將白色粉圓加入奶茶。無論來源，珍珠奶茶已成為臺灣最具代表性的飲料，風靡全球。全球珍珠奶茶市場規模超過2000億新臺幣。"
    ))
    records.append(qa(
        "What are the must-try foods in Taiwan?",
        "Must-try Taiwanese foods include: **beef noodle soup** (紅燒牛肉麵), **braised pork rice** (滷肉飯), **oyster omelet** (蚵仔煎), **stinky tofu** (臭豆腐, fried fermented tofu), **bubble milk tea** (珍珠奶茶), **Taiwanese fried chicken** (雞排), **sausage in sticky rice** (大腸包小腸), **three-cup chicken** (三杯雞), **guabao** (刈包, pork belly bun), and **shaved ice mountain** (刨冰) topped with fresh fruits."
    ))
    records.append(qa(
        "臺灣的臭豆腐為什麼叫臭豆腐？",
        "臭豆腐的「臭」來自發酵過程中產生的氣味。製作方式為將豆腐浸泡在由莧菜、芥菜、竹筍等蔬菜發酵而成的滷水中，經過數週至數月的發酵。發酵後的豆腐帶有強烈氣味，但油炸後外酥內嫩，搭配酸甜泡菜與醬油膏食用，風味獨特。臺灣臭豆腐有油炸與清蒸兩種吃法。深坑（新北）為臭豆腐知名產地。"
    ))

    # --- Social issues ---
    records.append(qa(
        "臺灣面臨哪些主要的社會問題？",
        "臺灣當前主要社會問題包括：人口少子化與高齡化（2025年進入超高齡社會，65歲以上占20%+）、低薪與房價高漲（青年購屋負擔沉重）、城鄉差距（西部與東部發展不均）、環境污染（PM2.5空污、塑膠廢棄物）、核廢料處理爭議、外籍移工權益問題、性別平等（雖已亞洲領先但仍待進步）。"
    ))
    records.append(qa(
        "臺灣的高齡化社會現狀如何？",
        "臺灣於2025年正式邁入超高齡社會（65歲以上人口占比超過20%），老年人口約480萬人。扶養比持續上升，勞動力人口逐年減少。長照2.0政策提供居家服務、日間照顧與機構住宿式服務。銀髮產業（醫療保健、輔具、智慧居家）快速發展。延後退休年齡（2024年起年金改革漸進延後至65歲）為重要政策方向。"
    ))
    records.append(qa(
        "臺灣的外籍移工政策是怎樣的？",
        "臺灣外籍移工人數約80萬人（2025年數據），主要來自印尼、越南、菲律賓、泰國，從事製造業、營造業、看護工與漁工。移工人數逐年增加，對臺灣經濟與長照體系貢獻顯著。但移工面臨勞動條件不佳、仲介費過高等問題。政府推動《最低服務年限》改革與1955移工專線。部分團體呼籲建立仲介免費化制度。"
    ))

    # =========================================================================
    # SECTION 4: NTNU SPECIFIC - DEEPER DETAIL (~70 records)
    # =========================================================================

    # --- NTNU's role in Taiwan's education reform ---
    records.append(qa(
        "師大在臺灣的教育改革中扮演什麼角色？",
        "師大作為臺灣師資培育的龍頭，歷來參與多次重大教育改革。包括九年國民義務教育（1968年，師大負責大量國中師資培訓）、九年一貫課程改革（2001年，師大教授主導課程綱要設計）、十二年國民基本教育（2014年，師大負責教師增能與課程轉化）。師大教育學院長期為教育部政策智庫，各領域教材教法研究亦由師大教授領銜。"
    ))
    records.append(qa(
        "師大對臺灣師資培育制度的影響？",
        "師大是臺灣師資培育的核心機構，早期（1946–1994）為全臺中學教師主要培育管道。1994年《師資培育法》修正開放一般大學設教育學程後，師大轉型為綜合型大學，但仍維持師資培育特色。師大設立師資培育學院、師資培育與就業輔導處，每年培育大量合格教師。師大實習學校網絡遍佈全國各縣市，遠距教師增能課程影響範圍更擴及離島。"
    ))
    records.append(qa(
        "什麼是師大的教育實習制度？",
        "師大教育實習為師資培育課程的核心環節，學生修畢教育學程後需至中等學校進行為期半年的全時教育實習（2022年起改為半年）。實習期間由師大實習指導教授與學校輔導教師共同指導。實習成績及格後參加教師資格考試，通過後取得教師證書。師大與全臺超過300所中學簽約為實習合作學校。"
    ))
    records.append(qa(
        "How has NTNU influenced education policy in Taiwan?",
        "NTNU's College of Education has long served as a policy think tank for Taiwan's Ministry of Education. NTNU faculty have led curriculum framework design for the 9-Year Integrated Curriculum (2001) and 12-Year Basic Education (2014). The university's Research Center for Educational Policy and Evaluation conducts ongoing policy assessment. NTNU's teacher training programs have shaped over 70% of Taiwan's secondary school teachers over the decades."
    ))

    # --- Detailed department history ---
    records.append(qa(
        "師大國文學系的歷史沿革？",
        "師大國文學系創立於1946年，為師大最早成立的學系之一。著名學者如林尹、高明、許世瑛、潘重規、鄭騫、王國瓔、曾永義等曾任教於此。系所發展涵蓋中國文學、臺灣文學、華語文教學等領域。圖書館古籍特藏豐富，為臺灣中國文學研究重鎮。每年舉辦「紅樓現代文學獎」與「全國高中生文藝營」，推動文學創作。"
    ))
    records.append(qa(
        "師大英語學系和梁實秋有什麼淵源？",
        "師大英語學系（原名英語研究所、英語系）為師大最早成立的學系之一，文學大師梁實秋於1949年來臺後長期任教於此（直至1974年退休），對系所發展影響深遠。梁實秋在此翻譯《莎士比亞全集》，完成其最重要的學術貢獻。師大設有梁實秋故居（和平校區），並由中文系主辦「梁實秋文學大師獎」，延續其文學精神。"
    ))
    records.append(qa(
        "師大歷史學系有哪些研究特色？",
        "師大歷史學系成立於1962年，研究涵蓋中國史、臺灣史、世界史三大領域。臺灣史研究為重點特色，與臺灣史研究所合作密集。系所設有歷史學系數位人文實驗室，運用數位技術進行史料分析。著名師資包括王爾敏、李國祁、黃富三等學者。出版《臺灣師大歷史學報》為THCI核心期刊。"
    ))
    records.append(qa(
        "師大化學系和李遠哲有什麼關係？",
        "諾貝爾化學獎得主李遠哲於1959年畢業於師大化學系，後赴美深造，於1986年以交叉分子束實驗方法獲諾貝爾化學獎，為首位臺灣出生的諾貝爾獎得主。李遠哲對母校師大有深厚情感，曾多次返校演講並捐贈研究設備。師大化學系設有「李遠哲院士紀念講座」，激勵後進學子。他亦在前總統任內推動教育改革。"
    ))
    records.append(qa(
        "師大美術學系的發展歷程？",
        "師大美術學系成立於1947年，為臺灣最早設立之高等美術教育學系。黃君璧、溥心畬、張大千等渡海名家曾在此教學，奠定臺灣美術教育根基。系所培育出廖修平、陳銀輝、江明賢、袁金塔等知名藝術家。師大美術館（2021年啟用）為臺灣首座大學附屬美術館，舉辦國內外重要展覽。每年系展與畢業美展為臺灣藝壇盛事。"
    ))

    # --- NTNU's COVID-19 response ---
    records.append(qa(
        "師大在COVID-19疫情期間採取了哪些措施？",
        "COVID-19疫情期間（2020–2023年），師大迅速推出多項應變措施：全面實施遠距教學（同步與非同步課程）、建置雲端學習平臺（Moodle升級）、設置防疫專區與健康回報系統、校園出入口體溫量測與實聯制、暫停實體大型活動（畢業典禮改線上）、宿舍防疫管理、提供經濟弱勢學生緊急紓困金。師大亦開發線上華語課程，維持國際學生招生與教學。"
    ))
    records.append(qa(
        "師大在後疫情時代的數位教育轉型成果？",
        "疫情加速師大數位教育轉型：2022年成立網路大學辦公室，專責全球校園與數位學習發展；大規模開設線上課程與MOOC（磨課師），華語文數位課程尤其受到海外歡迎；教師培訓加強數位教學能力（EMI與遠距教學工作坊）；建置「師大數位學習平臺」整合課程資源。後疫情時代師大維持混合彈性教學模式，並將線上教學經驗融入常規課程。"
    ))

    # --- Green campus initiatives ---
    records.append(qa(
        "師大有什麼綠色校園政策和措施？",
        "師大推動多項綠色校園政策：綠建築校舍（公館校區新建工程採綠建築標章）、校園植栽多樣化與生態池設置、垃圾分類與資源回收系統、節能減碳措施（LED照明汰換、太陽能光電板設置、空調節能管理）、YouBike站點設置鼓勵低碳交通。師大環境教育中心負責規劃與推動綠色校園教育與永續發展。"
    ))
    records.append(qa(
        "師大在環境教育方面有什麼貢獻？",
        "師大環境教育中心為全臺最早成立的大學環境教育機構之一，從事環境教育師資培訓、課程開發與社會推廣。師大參與聯合國教科文組織永續發展教育（ESD）相關計畫，推動學校永續發展教育。師大STARS永續評比獲亞洲第二面金質獎章（2022年），在溫室氣體減量、水資源管理、校園生態多樣性等方面表現優異。"
    ))
    records.append(qa(
        "師大有哪些永續發展SDGs相關成果？",
        "師大在THE Impact Rankings（泰晤士高等教育影響力排名）中，SDG 5（性別平等）、SDG 6（潔淨水與衛生）、SDG 10（減少不平等）等項目表現突出。校內設有永續發展推動委員會，將SDGs融入教學、研究與校務治理。2022年STARS評比獲金質獎章，為亞洲第二校獲得此殊榮。師大USR（大學社會責任）計畫亦聚焦永續城鄉與教育品質。"
    ))

    # --- Student exchange programs ---
    records.append(qa(
        "師大有哪些國際交換學生計畫？",
        "師大與全球超過200所大學簽訂姊妹校協定，提供廣泛的交換學生計畫。熱門交換學校包括日本東京大學、京都大學、韓國首爾大學、美國加州大學系統、德國柏林洪堡大學、法國里昂第三大學等。每年約300名師大學生出國交換，同時約400名境外學生來師大交換。師大國際事務處提供獎學金（如「師大飛鷹計畫」）與行前輔導。"
    ))
    records.append(qa(
        "師大國語教學中心的歷史和規模？",
        "師大國語教學中心（Mandarin Training Center, MTC）成立於1956年，為全球歷史最悠久、規模最大的華語教學機構之一。每年招收來自100多國的學生超過4,000人次，提供從初級到高級的華語課程。中心位於和平校區博愛樓，師資超過100人。MTC開發的《當代中文課程》系列教材被全球廣泛使用。TOCFL（華語文能力測驗）亦由師大華語文與科技研究中心主導。"
    ))
    records.append(qa(
        "師大的國際學生宿舍情況如何？",
        "師大為國際學生提供多種住宿選擇：校本部附近的「學人一舍」、「學人二舍」與「國際學舍」等，生活機能便利（近師大商圈、捷運站）。國際學生享有優先住宿權。宿舍類型包括單人房、雙人房與四人房，配備基本家具、網路與公共衛浴。住宿費用依房型約每月新臺幣5,000–12,000元。林口校區亦提供僑生宿舍。"
    ))
    records.append(qa(
        "What international exchange programs does NTNU offer?",
        "NTNU has partnership agreements with over 200 universities worldwide, including top institutions in Japan (University of Tokyo), Korea (Seoul National University), the US (UC system), and Europe (Humboldt University Berlin, Lyon III). Each year approximately 300 NTNU students go abroad while 400 international students come to NTNU. The university offers scholarships through programs like the 'NTNU Eagle Project' and provides pre-departure orientation and academic counseling."
    ))

    # --- Community relationship ---
    records.append(qa(
        "師大與所在地大安區的關係如何？",
        "師大和平校區位於大安區核心地帶，與在地社區關係密切。「師大夜市」（師大路商圈）因師大學生消費而形成，但近年因居民反映噪音與油煙問題，師大與社區合作推動「師大商圈轉型」計畫，輔導店家符合環保與噪音規範。大安區居民可使用師大運動場館（收費制）與圖書館。師大USR計畫亦在大安區辦理社區教育課程與藝文活動。"
    ))
    records.append(qa(
        "師大在USR大學社會責任方面做了哪些事？",
        "師大積極推動USR（大學社會責任）計畫，重點項目包括：社區教育（大安區、林口區課後輔導與終身學習）、偏鄉教育（數位學伴計畫，協助偏鄉學童學習）、環境教育（校園與社區環境教育推廣）、文化保存（協助地方文史調查與保存）、新住民支持（提供華語教育與社會資源轉介）。多個USR計畫獲得教育部補助，並與臺北市、新北市政府合作。"
    ))

    # --- Sports history ---
    records.append(qa(
        "師大體育系的歷史和成就？",
        "師大體育與運動科學系為臺灣最早成立的體育學系之一，1950年代即開始培育體育師資與研究人才。師大在各大專體育賽事成績斐然：男子排球隊曾獲隊史第24座全國冠軍、女子桌球與羽球亦為傳統強項。師大培育出多位奧運選手與國手級教練。體育表演會自1950年代開始舉辦，為師大最具傳承的校園活動之一。"
    ))
    records.append(qa(
        "師大運動競技學系有哪些特色？",
        "師大運動競技學系成立於2002年，專注優秀運動員的培育與發展。系上學生多為各運動項目的國手或潛力選手，接受學科與術科雙軌訓練。師大提供運動獎學金與課業輔導，協助選手兼顧學業與訓練。畢業生可投入職業運動、國家級教練、體育行政等領域。師大男排選手多次入選中華隊，為大專排球聯賽常勝軍。"
    ))
    records.append(qa(
        "師大體育表演會的特色和歷史？",
        "師大體育表演會為師大年度重要傳統，每年初夏在和平校區體育館舉行。節目由運動與休閒學院畢業班學生自主編排、練習與演出，融合競技運動、舞蹈、體操、特技與創意表演。傳統可追溯至1950年代的「體育實習表演」，至今已超過70年歷史，是師大最具傳承與代表性的學生自主活動之一。"
    ))

    # --- Publications and journals ---
    records.append(qa(
        "師大出版哪些學術期刊？",
        "師大出版多種學術期刊，涵蓋教育、人文、科學等領域，重要期刊包括：《教育研究集刊》（TSSCI）、《師大學報》（含教育類、人文與社會科學類、數理與科技類、科學教育類）、《臺灣師大歷史學報》（THCI）、《同心圓：語言學研究》（Concentric, SSCI收錄）、《體育學報》（TSSCI）、《圖書資訊學刊》、《華語文教學研究》。均採嚴謹審稿制度，學術影響力卓越。"
    ))
    records.append(qa(
        "《師大學報》的歷史和涵蓋範圍？",
        "《師大學報》（Journal of National Taiwan Normal University）創刊於1956年，為師大歷史最悠久的學術期刊。原為綜合性學報，2007年起分為教育類、人文與社會科學類、數理與科技類、科學教育類四版。採雙匿名審稿制，收錄於TSSCI、THCI等學術資料庫。每年穩定出刊，是臺灣師範大學體系最具代表性的學術期刊。"
    ))

    # --- Innovation and startup ecosystem ---
    records.append(qa(
        "師大的創新創業生態系統是怎樣的？",
        "師大推動創新創業的機制包括：師大育成中心（提供新創團隊進駐空間與輔導）、創業學程（管理學院開設）、跨域科技產業創新研究學院（AI與綠能科技產學合作）、師大產學合作中心（媒合企業與研究團隊）。師大每年舉辦創新創業競賽，鼓勵學生將研究成果轉化為商業模式。華語文科技、教育科技（EdTech）、藝術設計為師大特色的新創領域。"
    ))
    records.append(qa(
        "師大育成中心輔導哪些類型的新創公司？",
        "師大育成中心進駐的新創團隊以教育科技、文化創意、華語文數位學習、綠色能源、社會企業為主要領域。育成中心提供進駐空間、商業輔導、業師媒合、資金鏈結（天使投資與政府補助申請輔導）等服務。成功案例包括華語文線上教學平臺、特殊教育輔具開發公司、文化資產數位保存工作室等。中心亦與臺大系統育成網絡合作。"
    ))
    records.append(qa(
        "師大跨域科技產業創新研究學院有哪些研究所？",
        "跨域科技產業創新研究學院成立於2023年，設有「AI跨域應用研究所」與「綠能科技與永續治理研究所」。前者聚焦人工智慧在不同領域的應用研究，後者結合能源科技、環境永續與政策治理。兩個研究所皆強調產學合作，與國內外企業共同進行實務研究，培育產業所需的高階跨域人才。"
    ))
    records.append(qa(
        "What is NTNU's role in Taiwan's EdTech industry?",
        "NTNU has a strong EdTech ecosystem through its College of Education, the Learning Informatics Professional College, and the Chinese Language and Technology Center. The university develops digital learning platforms, assessment tools (including TOCFL), and adaptive learning systems. NTNU's EdTech startups focus on language learning technology, special education tools, and online assessment systems. The university's NTNU NEXT program also explores new models for high school-university articulation through digital learning."
    ))

    # --- Additional NTNU depth ---
    records.append(qa(
        "師大在人文藝術方面的研究和教學有什麼特色？",
        "師大在人文藝術領域底蘊深厚：文學院下設國文、英語、歷史、地理、臺灣語文等系，並有全球華文寫作中心。藝術學院設有美術、設計、音樂等系，師大美術館為大學附屬美術館之先驅。音樂學院擁有全臺唯一音樂圖書館，收藏大量樂譜與音樂文獻。師大藝術節、音樂節與人文季提供密集的展演與講座。"
    ))
    records.append(qa(
        "嘉南平原是臺灣最大的平原嗎？",
        "嘉南平原是臺灣最大的平原，面積約4,550平方公里，北起彰化、南至高雄，涵蓋雲林、嘉義、臺南等農業重鎮。平原由濁水溪、北港溪、八掌溪、曾文溪等河流沖積而成。為臺灣最重要的農業區，生產稻米、甘蔗、雜糧與蔬菜。嘉南大圳（烏山頭水庫）灌溉系統支撐此平原的農業生產。"
    ))
    records.append(qa(
        "臺灣有哪些知名水庫？",
        "臺灣知名水庫包括：曾文水庫（最大，容量約5億m³，嘉義）、翡翠水庫（供應大臺北地區，新北）、石門水庫（桃園，供應北部用水）、德基水庫（大甲溪，水力發電為主，臺中）、烏山頭水庫（嘉南大圳蓄水，臺南）。多數水庫兼具供水、發電、防洪與觀光功能。"
    ))
    records.append(qa(
        "What is Taiwan's climate like by region?",
        "Taiwan's climate varies by region. Northern Taiwan (Taipei, Keelung) has a subtropical climate with rain year-round and cooler winters. Central Taiwan (Taichung) enjoys the best weather with more sunshine and less rain. Southern Taiwan (Kaohsiung, Tainan) is tropical — hot year-round with a distinct dry winter and rainy summer. Eastern Taiwan (Hualien, Taitung) has a climate similar to the north but with more typhoon impact. Mountain areas have a temperate to alpine climate. The best all-around weather is in central and southern Taiwan during autumn."
    ))
    records.append(qa(
        "宜蘭為什麼多雨？",
        "宜蘭位於臺灣東北部，為東北季風的第一道迎風面。每年10月至次年3月，東北季風挾帶水氣從太平洋吹向臺灣，在宜蘭遭遇雪山山脈抬升，形成地形雨，導致冬季連綿陰雨。宜蘭年均降雨日約200天，年降雨量超過2,500毫米。蘇澳地區更因特殊地形，降雨量尤高。"
    ))
    records.append(qa(
        "淡水的歷史地位是什麼？",
        "淡水位於淡水河口北岸，為臺灣北部最早開發的港口之一。1629年西班牙人建立聖多明哥城（今紅毛城），1642年荷蘭人驅逐西班牙人。19世紀淡水成為國際商港（1858年天津條約開港），茶葉、樟腦大量出口。淡水老街、紅毛城、真理大學（前身為牛津學堂）等見證歷史。現為新北市熱門觀光區，以淡水夕照聞名。"
    ))
    records.append(qa(
        "Why is Hualien known for marble?",
        "Hualien's marble industry stems from the region's unique geology. The Central Mountain Range in Hualien contains massive deposits of high-quality marble, formed from limestone metamorphosed by tectonic pressure millions of years ago. The Liwu River carved through these marble formations to create Taroko Gorge. During the 1960s–90s, Hualien was Taiwan's marble-processing capital, with factories producing construction materials, sculptures, and decorative items. The industry has declined due to environmental concerns and import competition."
    ))
    records.append(qa(
        "臺東的熱氣球嘉年華是什麼活動？",
        "臺東熱氣球嘉年華（臺灣國際熱氣球嘉年華）每年6–8月在鹿野高臺舉行，為期約45天，是臺灣最具代表性的熱氣球活動。來自全球數十個國家的熱氣球飛行員參與，每天清晨與傍晚進行自由飛行與繫留體驗。活動包含熱氣球光雕秀、草原音樂會與周邊市集。每年吸引超過百萬人次造訪臺東，帶動東部觀光。"
    ))
    records.append(qa(
        "臺灣有哪些特殊的海岸地形？",
        "臺灣海岸地形多樣：北海岸（野柳、石門）為侵蝕海岸，有女王頭、燭臺石等蕈狀岩；東海岸（清水斷崖、石梯坪）為斷層海岸，陡峭壯麗；西海岸（高美濕地、七股潟湖）為堆積海岸，沙灘與濕地廣闊；南海岸（墾丁、佳樂水）為珊瑚礁海岸。野柳地質公園的女王頭因風化持續變細，為保育重點。"
    ))
    records.append(qa(
        "臺灣的水資源狀況如何？",
        "臺灣年降雨量約2,500毫米（為世界平均的三倍），但因地形陡峭、河川短促，大部分雨水迅速流入海洋，水資源利用率僅約20%。水庫容量有限且淤積嚴重。每年枯水期（11–4月）中南部常面臨缺水危機。2021年百年大旱即為典型案例。政府推動再生水、海水淡化、節水農業等因應措施。"
    ))
    records.append(qa(
        "日月潭萬人泳渡活動的歷史？",
        "日月潭萬人泳渡活動始於1983年，每年中秋節前後舉行，全程約3公里（從朝霧碼頭到伊達邵碼頭）。每年吸引超過2萬名國內外泳者參加，為金氏世界紀錄認證的「最大規模水上泳渡活動」。泳渡日月潭被視為臺灣人「一生必做的三件事」之一（另兩件為登玉山、環島）。南投縣政府主辦，兼具運動觀光效益。"
    ))
    records.append(qa(
        "臺灣有哪些重要的濕地？",
        "臺灣重要濕地包括：高美濕地（臺中，以夕陽與風車聞名，保育招潮蟹與彈塗魚）、七股潟湖（臺南，養蚵與紅樹林生態）、四草濕地（臺南，綠色隧道觀光船）、鰲鼓濕地（嘉義，候鳥重要棲息地）、關渡自然公園（臺北，候鳥觀察熱點）。臺灣濕地生態豐富，為東亞候鳥遷徙路線的重要中繼站。"
    ))
    records.append(qa(
        "臺灣的防災體系是怎樣的？",
        "臺灣防災體系分三層級：中央（災害防救會報，行政院）、縣市（縣市災害應變中心）、鄉鎮（基層防災單位）。針對颱風、地震、水災、土石流等自然災害，設有中央氣象局（預警）、國家災害防救科技中心（NCDR，技術支援）、消防署（緊急救援）。全民防災簡訊、細胞廣播（PWS）為重要預警工具。1999年921地震後大幅強化防災體系。"
    ))
    records.append(qa(
        "澎湖花火節是怎麼開始的？",
        "澎湖花火節始於2003年，最初為彌補SARS疫情對澎湖觀光的衝擊而舉辦。每年4–6月在馬公市觀音亭園區施放煙火，搭配專業音樂演出。後發展為澎湖年度最重要觀光活動，每年吸引約30–50萬遊客。花火節結合澎湖玄武岩景觀、海鮮美食與海上活動，帶動離島經濟。2020年後加入無人機燈光秀增加看點。"
    ))
    records.append(qa(
        "臺灣有什麼必訪的森林遊樂區？",
        "臺灣必訪森林遊樂區包括：溪頭（南投，大學池、空中走廊，柳杉林）、阿里山（嘉義，神木、雲海、森林鐵路）、太平山（宜蘭，蹦蹦車、翠峰湖）、大雪山（臺中，臺灣鐵杉林、野生動物觀察）、墾丁（屏東，熱帶海岸林）、池南（花蓮，哈崙森林鐵道歷史）。均由林業保育署管理，設有步道與住宿設施。"
    ))
    records.append(qa(
        "臺灣西部沿海有哪些重要工業區？",
        "臺灣西部沿海工業區密集，主要為：高雄臨海工業區（中鋼、中油、造船）、雲林麥寮六輕（臺塑石化園區，產值驚人）、臺中港關連工業區（機械、金屬製品）、彰濱工業區（鹿港，風力發電與製造業）、臺南科技工業區（光電、半導體）。形成從桃園到高雄的西部工業走廊，支撐臺灣製造業出口。"
    ))
    records.append(qa(
        "臺灣的自行車文化有多發達？",
        "臺灣是全球自行車製造重鎮，捷安特（Giant，全球最大自行車品牌）、美利達（Merida）均為臺灣品牌。環島自行車道（環島1號線，全長約968公里）為熱門活動，每年數萬人完成自行車環島。河濱自行車道遍布各大城市（臺北、新北、臺中、高雄）。YouBike公共自行車系統自2009年啟動，現已擴展至全臺多個縣市，站點數千、車輛數萬。"
    ))
    records.append(qa(
        "臺灣的海洋資源與漁業情況？",
        "臺灣遠洋漁業發達，為全球前三大遠洋漁業國之一，作業範圍涵蓋太平洋、印度洋與大西洋。鮪魚、魷魚、秋刀魚捕撈量居世界前列。近海漁業則以東部海域（黑潮帶來豐富漁類）與澎湖海域為主。養殖漁業（石斑魚、虱目魚、鰻魚、臺灣鯛）亦具規模。但漁業面臨過漁、越界捕撈爭議與漁工權益等問題。"
    ))
    records.append(qa(
        "臺灣的水上活動有哪些熱門地點？",
        "臺灣水上活動地點豐富：衝浪（宜蘭烏石港、臺東金樽、屏東佳樂水）、潛水（綠島、蘭嶼、墾丁後壁湖、澎湖）、SUP立槳（日月潭、花蓮清水斷崖、澎湖）、獨木舟（花蓮清水斷崖、東北角龍洞、小琉球）、泛舟（花蓮秀姑巒溪）、輕艇（冬山河、愛河）。全年均可從事水上活動，夏季尤為旺季。"
    ))
    records.append(qa(
        "臺灣的鐵路歷史是怎樣的？",
        "臺灣鐵路始於清代劉銘傳1887年興建的基隆–新竹段（1893年通車）。日治時期完成縱貫線（基隆–高雄，1908年通車），並興建宜蘭線、花東線等支線。1979年鐵路電氣化完成，1991年南迴鐵路通車形成環島鐵路網。2007年臺灣高鐵通車（臺北–高雄90分鐘）。阿里山森林鐵路為世界級登山鐵路。臺鐵太魯閣號、普悠瑪號為傾斜式列車，提升東部運輸效率。"
    ))
    records.append(qa(
        "What were the major Dutch contributions to Taiwan?",
        "The Dutch East India Company (VOC) ruled Taiwan from 1624 to 1662. Their major contributions include: (1) Building Fort Zeelandia (modern-day Anping, Tainan) as their administrative center; (2) Introducing large-scale Han Chinese immigration and sugarcane cultivation; (3) Establishing the first formal tax system and land registration; (4) Missionaries creating written records of Siraya language using Roman script (Sinckan Manuscripts); (5) Introducing cattle and plow farming to indigenous communities. The Dutch legacy is most visible in Tainan's historical sites."
    ))
    records.append(qa(
        "清領時期臺灣的移民社會是怎樣的？",
        "清領初期嚴禁漢人攜眷來臺（渡臺禁令，1684–1875年），導致臺灣社會以單身男性為主，形成「羅漢腳」現象（單身遊民）。分類械鬥頻繁（漳泉械鬥、閩客械鬥），族群對立嚴重。19世紀後渡臺禁令鬆綁，社會漸趨穩定。移民按原鄉祖籍形成聚落，出現郊商（貿易組織，如臺南三郊）與宗親會。墾首制（業主與佃戶關係）主導土地開發。"
    ))
    records.append(qa(
        "日治時期臺灣的現代化建設有哪些？",
        "日治時期現代化建設涵蓋多層面：交通（基隆高雄港現代化、縱貫鐵路、嘉義–阿里山森林鐵路）、水利（嘉南大圳、桃園大圳、瑠公圳）、電力（日月潭水力發電所、明潭發電廠）、都市建設（棋盤式道路、下水道、自來水、公園）、公共衛生（防治傳染病、衛生所體系）、郵電（郵政與電信網絡）。這些建設奠定臺灣現代化基礎。"
    ))
    records.append(qa(
        "美麗島事件對臺灣民主運動的影響？",
        "美麗島事件（1979年12月10日）雖遭國民黨大規模鎮壓，但反而激化臺灣民主運動。審判過程中，參與者家屬與辯護律師（包括陳水扁、蘇貞昌、謝長廷等）透過媒體將民主理念傳播至全臺。1986年民進黨成立，核心成員多為美麗島事件參與者。事件被視為臺灣民主化的重要轉捩點，直接促使1987年解嚴。"
    ))
    records.append(qa(
        "臺灣的政黨政治發展歷程？",
        "臺灣政黨政治發展始於1986年民進黨突破黨禁創黨（比解嚴早一年）。1987年解嚴後逐步開放黨禁與報禁。2000年首次政黨輪替，民進黨陳水扁當選總統，打破國民黨長期執政。2008年國民黨馬英九重新執政。2016年民進黨蔡英文再次輪替。2024年民進黨賴清德當選總統，延續執政。第三勢力有民眾黨（柯文哲創立，2019年）、時代力量（2015年）等政黨。"
    ))
    records.append(qa(
        "日本殖民統治對臺灣文化產生了哪些影響？",
        "日治時期對臺灣文化影響深遠：語言方面，日語詞彙融入臺語（如「便當」、「歐巴桑」、「阿莎力」）；建築方面，西洋古典、現代主義與和風建築遍佈全臺（總統府、臺中車站、監察院）；飲食方面，引入生魚片、味噌湯、咖哩等並本土化；生活習慣，公共浴場、榻榻米、和服等影響日常；教育方面，建立近代學校體系（師大、臺大均有日治淵源）。"
    ))
    records.append(qa(
        "臺灣的白色恐怖時期有多少政治犯？",
        "根據國家發展委員會檔案管理局統計，1949–1991年間因政治案件遭起訴者約10萬人，其中約4,000–5,000人遭處決，約8,000人被判處無期徒刑或長期監禁。綠島、小琉球、蘭嶼曾設有政治犯監獄。促進轉型正義條例（2017年）通過後，政府推動開放政治檔案、撤銷有罪判決與賠償。白色恐怖對臺灣社會的創傷影響至今。"
    ))
    records.append(qa(
        "臺灣的媒體自由度如何？",
        "臺灣長期在國際新聞自由度評比中名列亞洲前茅（2024年無國界記者組織排名第27名，亞洲僅次於日本）。1987年解嚴後報禁解除，媒體蓬勃發展。現有主要報紙（蘋果、自由、聯合、中時）、電視臺（臺視、中視、華視、民視、公視）、網路媒體（報導者、端傳媒、關鍵評論網）。媒體面臨財團化、政治立場鮮明、假訊息等挑戰。"
    ))
    records.append(qa(
        "What was Taiwan's role in the Cold War?",
        "During the Cold War (1947–1991), Taiwan occupied a strategic position as the front line of the US containment policy against communism in Asia. The US provided significant military and economic aid to Taiwan (totaling about $4 billion from 1951–1965). Taiwan served as a CIA operations base for covert actions in mainland China. After the US shifted recognition to the PRC in 1979, the Taiwan Relations Act maintained unofficial US-Taiwan defense ties. Taiwan also became a showcase of capitalist development, contrasting with mainland China's communist economy."
    ))
    records.append(qa(
        "臺灣的第三勢力有哪些主要政黨？",
        "臺灣第三勢力主要政黨包括：臺灣民眾黨（TPP，2019年由柯文哲創立，以務實、科學、理性為訴求）、時代力量（NPP，2015年由太陽花學運世代成立，偏進步派）、臺灣基進（立場鮮明，主張臺灣獨立建國）、親民黨（2000年由宋楚瑜創立，後漸式微）、新黨（1993年創立，偏統派）。第三勢力在立法院席次有限，但在特定議題上有關鍵影響力。"
    ))
    records.append(qa(
        "臺灣的選舉制度是怎樣的？",
        "臺灣總統副總統由人民直接選舉（相對多數決），任期四年，連選得連任一次。立法委員（立法院）共113席：區域立委73席（單一選區相對多數決）、原住民立委6席（山地與平地各3席）、不分區立委34席（政黨比例代表制，須獲5%以上選票）。縣市長、縣市議員、鄉鎮市長等地方公職人員亦由選舉產生。全國性選舉通常每兩年舉行一次。"
    ))
    records.append(qa(
        "臺灣有哪些重要的文化節慶活動？",
        "臺灣年度重要節慶包括：春節（1–2月，全家團圓）、元宵節（平溪天燈、鹽水蜂炮、炸寒單爺）、大甲媽祖遶境（3–4月，9天8夜徒步）、清明節掃墓、端午節（龍舟賽、粽子）、七夕（情人節）、中元節（普渡、搶孤）、中秋節（烤肉、月餅、柚子）、國慶日（10月10日，總統府前慶典）、各地豐年祭（7–8月，原住民族）。"
    ))
    records.append(qa(
        "臺灣的殯葬文化有什麼特色？",
        "臺灣殯葬文化融合儒家、道教、佛教與民間信仰。傳統儀式包括：燒庫錢（往生紙錢）、師公（道士）誦經、五子哭墓（專業哭喪）、孝女白琴（哭調仔）、頭七到滿七的祭拜。近年火葬率超過95%（因土地有限），納骨塔（靈骨塔）普及。環保自然葬（樹葬、花葬、海葬）逐漸推廣。喪禮中的電子花車與孝女哭墓為臺灣特色。"
    ))
    records.append(qa(
        "臺灣的宗教自由狀況如何？",
        "臺灣憲法保障宗教自由，為亞洲宗教最自由的社會之一。佛教、道教、民間信仰、基督教、天主教、一貫道、伊斯蘭教等和平共存。寺廟與教堂密度極高（約1.5萬間寺廟、3千間教堂）。廟會遶境、佛誕節（浴佛）、聖誕節等各宗教節日均公開慶祝。多元宗教融合的現象常見，如佛道不分、民間信仰吸收儒釋道元素。"
    ))
    records.append(qa(
        "臺灣的元宵節有哪些特殊活動？",
        "臺灣元宵節（農曆正月十五）活動豐富多元：臺灣燈會（由交通部主辦，每年在不同城市，大型主燈秀）；平溪天燈節（新北，數千盞天燈同時升空）；鹽水蜂炮（臺南，炮城射擊，刺激的炮陣體驗）；炸寒單爺（臺東，肉身承受鞭炮轟炸，為寒單爺除穢）；苗栗火旁龍（舞龍與鞭炮結合）。各地廟宇舉辦猜燈謎、提燈籠活動。"
    ))
    records.append(qa(
        "What are Taiwan's most distinctive festivals?",
        "Taiwan's most distinctive festivals include: (1) **Pingxi Sky Lantern Festival** — thousands of lanterns released into the night sky in New Taipei City; (2) **Yanshui Beehive Fireworks Festival** — participants wear helmets while being 'bombarded' by rocket-like fireworks in Tainan; (3) **Dajia Mazu Pilgrimage** — an 8-day, 300km walking procession from Taichung to Chiayi; (4) **Taitung Bombing of Handan Ye** — a deity carried through crowds as firecrackers are thrown at it; (5) **Penghu Fireworks Festival** — summer fireworks over the ocean."
    ))
    records.append(qa(
        "臺灣的茶葉外銷歷史？",
        "臺灣茶葉外銷始於19世紀，英國商人陶德（John Dodd）於1865年將烏龍茶引進臺北產製，1869年以「Formosa Tea」品牌出口美國，大受歡迎。1870年代臺灣茶葉躍居最大出口商品。日治時期茶葉持續為重要外匯來源，以包種茶與烏龍茶為主。戰後茶葉產業轉向內銷市場與精緻化。現代臺灣茶以高山茶為主流，外銷市場以日本、歐美為主。"
    ))
    records.append(qa(
        "臺灣聽障奧運的歷史？",
        "2009年臺北聽障奧運會（Summer Deaflympics）為臺灣首次舉辦的綜合性國際運動賽會，也是聽奧史上參賽人數最多的一屆（80餘國、約4,000名選手）。比賽場館包括臺北田徑場（開閉幕典禮）、松山運動中心等。臺灣選手獲得獎牌數不俗。聽奧後遺留的場館與志工文化成為重要資產，提升臺灣辦理國際賽事的經驗與信心。"
    ))
    records.append(qa(
        "臺灣的宗教活動中常見的陣頭是什麼？",
        "陣頭為臺灣廟會遶境中的表演隊伍，可分為「文陣」（音樂性表演，如車鼓陣、牛犁陣、採茶舞）與「武陣」（武術與特技表演，如八家將、官將首、宋江陣、獅陣、龍陣）。八家將為最受矚目的陣頭之一，扮演城隍爺的護衛，臉譜與步伐極具特色。陣頭文化傳統與現代衝突時有討論（如少年陣頭偏差行為、宗教商業化）。"
    ))
    records.append(qa(
        "臺灣有哪些傳統樂器？",
        "臺灣傳統樂器受閩南與客家影響：南管樂器（琵琶、洞簫、二弦、三弦、拍板）、北管樂器（嗩吶、鼓、鑼、鈸）、國樂（古箏、二胡、笛子、揚琴）。臺灣本土創新樂器包括：陳中申研發的「臺灣笛」、各種改良式嗩吶。原住民族傳統樂器有口簧琴（泰雅、布農）、弓琴（布農）、鼻笛（排灣）、木鼓（阿美）。客家八音為臺灣重要無形文化資產。"
    ))
    records.append(qa(
        "臺灣的書法與水墨畫傳統？",
        "書法與水墨畫為臺灣重要傳統藝術。渡海來臺的書畫家（溥心畬、黃君璧、張大千）在師大美術系教學，奠定學院派水墨根基。臺灣書法界有「全省美展書法部」與「橫山書法藝術館」（桃園）等展覽空間。水墨畫方面，現代水墨先驅劉國松打破傳統框限。民間書法教育普及，社區大學與寺廟書法班隨處可見，春聯書寫仍為年節習俗。"
    ))
    records.append(qa(
        "臺灣的布袋戲為什麼重要？",
        "臺灣布袋戲（掌中戲）為最具代表性的傳統偶戲藝術，發展出極具臺灣特色的表演形式。黃俊雄於1970年代將布袋戲搬上電視（雲州大儒俠史豔文），創下超過90%的收視率紀錄。霹靂布袋戲（黃強華、黃文擇兄弟）進一步發展出長篇系列，創造龐大IP生態（周邊商品、電影、遊戲）。雲林虎尾為布袋戲重鎮，設有布袋戲館。"
    ))
    records.append(qa(
        "臺灣有哪些無形文化資產？",
        "臺灣文化部公告的國家級無形文化資產包括：南管音樂、北管音樂、歌仔戲、布袋戲、皮影戲、傀儡戲、客家八音、魯凱族石板屋建造技術、賽夏族矮靈祭、太魯閣族織布工藝、排灣族刺繡、臺灣原住民族口述傳統等。各縣市政府亦登錄地方性無形文化資產，保存在地傳統技藝與知識。"
    ))
    records.append(qa(
        "臺灣的全民健保制度是怎樣的？",
        "臺灣全民健康保險（NHI）於1995年實施，為強制性社會保險制度，所有在臺居住滿6個月的國民均須參加。就醫只需出示健保卡，負擔掛號費與部分負擔。就醫便利性高、費用低（每月保費約新臺幣800–1,500元），涵蓋範圍包括西醫、中醫、牙醫、藥品、復健、住院等。健保滿意度長期超過80%，但面臨人口老化導致醫療支出攀升的財務挑戰。藥品部分負擔與分級醫療為改革方向。"
    ))
    records.append(qa(
        "What is the National Health Insurance in Taiwan?",
        "Taiwan's National Health Insurance (NHI), launched in 1995, is a universal single-payer healthcare system covering all residents. It offers comprehensive medical services — outpatient, inpatient, dental, Chinese medicine, prescription drugs, and rehabilitation — with low copayments and premiums (roughly NT$800–1,500 monthly). Citizens pay nothing at point of service with their NHI card, only a modest registration fee. The NHI is known for its efficiency, with over 80% public satisfaction and low administrative costs (about 2% of total healthcare spending)."
    ))
    records.append(qa(
        "臺灣的教育制度是怎樣的？",
        "臺灣教育制度包含：學前教育（幼兒園，2–6歲）、國民小學（6年，7–12歲）、國民中學（3年，13–15歲）、高級中等學校（3年，16–18歲，分普通型/技術型高中）、大學（4–7年，學士）、研究所（碩士1–4年、博士2–7年）。2014年實施十二年國民基本教育（高中職免學費、免試入學為主）。大專入學採多元管道（繁星推薦、個人申請、考試分發）。臺灣高等教育普及率超過70%。"
    ))
    records.append(qa(
        "臺灣的族群構成是怎樣的？",
        "臺灣族群依移民先後可分為：原住民族（約2.5%，16族，南島語系）、福佬人（Hō-ló，約70%，最早漢人移民）、客家人（約15–18%，粵東閩西移民）、外省人（約10–12%，1949年後隨國民政府來臺）、新住民（約3%，近年跨國婚姻）。族群關係在政治與文化上影響深遠，多元族群政策與本土化教育為當代重要議題。"
    ))
    records.append(qa(
        "臺灣面臨的少子化問題有多嚴重？",
        "臺灣少子化問題為全球最嚴重之一：2024年總生育率約0.87（每位婦女一生生育數），遠低於替代水準2.1。2020年臺灣人口首次出現負成長，全年出生人數低於死亡人數。高房價、低薪資、育兒成本高、性別平等不足為主要原因。政府推出育兒津貼（每月5,000–7,000元）、托育補助、育嬰留職停薪津貼等政策，但效果有限。預計2050年人口將降至約1,900萬人。"
    ))
    records.append(qa(
        "臺灣的同志權益與同婚現狀？",
        "臺灣於2019年5月17日通過《司法院釋字第七四八號解釋施行法》，成為亞洲第一個同性婚姻合法化的國家。截至2025年，已有超過1萬對同性伴侶登記結婚。同婚通過後臺灣社會包容度持續提升，同志遊行（每年10月）為亞洲最大規模。但跨國同婚與人工生殖等權利仍在推動中。臺灣的同志權益進展在亞洲具指標性意義。"
    ))
    records.append(qa(
        "師大教育學院的組織和特色？",
        "師大教育學院為全國歷史最悠久的教育學院，設有教育學系、教育心理與輔導學系、特殊教育學系、公民教育與活動領導學系、幼兒與家庭科學學系，以及學習資訊專業學院。學院設有教育研究與創新中心、心理與教育測驗研究發展中心（心測中心）等研究單位。院內教育類藏書量居全國之冠，TSSCI教育類期刊多由師大教育學院主編。"
    ))
    records.append(qa(
        "師大音樂學院有哪些學程與中心？",
        "師大音樂學院設有音樂學系（含西樂組、國樂組、聲樂組、鋼琴組、作曲組）、流行音樂產業碩士專班、亞洲流行音樂數位科技研究中心（AMP）。學院擁有全臺唯一音樂圖書館，收藏大量樂譜、CD與音樂文獻。音樂系館演奏廳定期舉辦師生音樂會與大師講座。學院亦與國家交響樂團（NSO）等機構合作提供實習機會。"
    ))
    records.append(qa(
        "師大公館校區有什麼新建設？",
        "公館校區為師大理工與社會科學重鎮，近年新建工程包括：科技與工程學院大樓（先進實驗室與教室）、理學院教學大樓更新。公館校區緊鄰臺大、臺科大，形成臺北南區大學城，三校學生可跨校修課與使用資源。校區內設有公館分館圖書館、學生餐廳與運動設施。捷運公館站步行約10分鐘，交通便利。"
    ))
    records.append(qa(
        "師大對於本土語言（臺語、客語、原民語）的教學與研究？",
        "師大在臺灣本土語言領域深耕多年：臺灣語文學系為全國少數以臺灣語文為名的學系，推動臺語文教育與研究。東亞學系與臺灣史研究所亦涉及本土語言相關研究。師大開設臺語、客語與原住民族語師資培訓課程，協助教育部推動本土語言教育。民族音樂研究所研究原住民族與客家音樂。師大亦參與本土語文教材編纂工作。"
    ))
    records.append(qa(
        "師大在COVID-19疫情期間的華語教學應對？",
        "COVID-19疫情期間，國語教學中心（MTC）迅速轉型為遠距華語教學，開發線上同步課程平臺與數位教材。海外無法來臺的學生可透過線上課程繼續學習華語。中心也推出華語夏令營線上版、華語教師線上培訓課程。疫情經驗促使MTC強化數位教學能力，後疫情時代持續提供混合教學模式，擴大服務全球華語學習者。"
    ))
    records.append(qa(
        "師大暑期有什麼特色營隊活動？",
        "師大每年暑期開辦多種營隊與課程：華語夏令營（國語教學中心，吸引全球學生來臺學華語）、高中暑期大學課程（NTNUNEXT，高中生預修大學學分）、兒童夏令營（進修推廣學院，科學、美術、音樂等主題）、體育夏令營（籃球、排球、游泳等）。各系所亦舉辦高中生營隊（中文營、歷史營、地理營、心理營等）吸引優秀學生。"
    ))
    records.append(qa(
        "師大與中研院有什麼合作關係？",
        "師大與中央研究院（中研院）維持密切合作：設立聯合博士學位學程（如農業生物科技學程）、合作研究計畫（天文物理、歷史語言、資訊科學等領域）、師生使用中研院圖書館與研究設備、中研院研究員在師大兼任教職。師大天文與重力研究中心與中研院天文所合作重力波研究。此合作模式擴展師大師生研究資源與國際視野。"
    ))
    records.append(qa(
        "What makes NTNU distinct among Taiwan's universities?",
        "NTNU is unique among Taiwan's universities for several reasons: (1) It is Taiwan's premier teacher-training institution with a 100+ year history tracing back to Japanese-era Taipei Higher School; (2) It houses the world's oldest and largest Mandarin Training Center (MTC, est. 1956), serving students from over 100 countries; (3) Its College of Education is the most comprehensive in Taiwan, housing the national high school exam development center; (4) NTNU is part of the NTU System, collaborating with NTU and Taiwan Tech; (5) It uniquely combines liberal arts, sciences, performing arts, and sports under one institution."
    ))
    records.append(qa(
        "師大有哪些企業合作與產學計畫？",
        "師大企業合作與產學計畫涵蓋多領域：與Google合作推動AI教育、與微軟合作數位學習平臺、與臺積電合作半導體人才培育、與華碩合作智慧校園、與國泰金控合作金融科技研究。師大產學合作中心每年媒合數十件產學合作案，技術轉移收入穩定成長。育成中心輔導的新創公司多位於教育科技與文化創意領域。"
    ))
    records.append(qa(
        "師大在聯合國永續發展目標（SDGs）的具體實踐？",
        "師大將SDGs融入校務發展與教學研究：SDG 4（優質教育）—師大師資培育與華語教育全球影響力；SDG 5（性別平等）—性平教育與愛洛生活節；SDG 6（潔淨水與衛生）—校園水資源管理與環境教育；SDG 10（減少不平等）—弱勢學生獎助學金與多元共融政策；SDG 11（永續城鄉）—USR在地實踐計畫；SDG 13（氣候行動）—校園碳中和規劃。2022年STARS金質獎章表彰師大在SDGs的貢獻。"
    ))
    records.append(qa(
        "師大的校友網絡如何？有哪些傑出校友？",
        "師大校友網絡龐大，遍布教育、學術、政治、文化、藝術、體育等領域。傑出校友包括：前總統府資政李國鼎（經濟推手）、諾貝爾獎得主李遠哲（化學系）、文學大師梁實秋（英語系教授，亦為校友）、知名作家簡媜（國文系）、藝人羅大佑（音樂系，雖未畢業）、籃球名將陳信安（運動競技系）、教育部長潘文忠（教育系）。師大校友總會與各縣市及海外分會保持活躍聯繫。"
    ))
    records.append(qa(
        "師大校園內有哪些學生餐廳和美食？",
        "師大和平校區內設有學生餐廳（俗稱「地餐」，即地下餐廳），提供自助餐、簡餐與麵食等平價餐點。師大路商圈（師大夜市）因師大學生而生，知名美食包括：師大鹽水雞、許記生煎包、馬來西亞咖哩、霖園牛奶大王。公館校區亦有學生餐廳與周邊餐飲（公館商圈緊鄰臺大）。林口校區則有學生餐廳等配套。師大生活便利通整合校園餐飲資訊。"
    ))
    records.append(qa(
        "師大有哪些促進學生國際移動力的計畫？",
        "師大推動多項計畫提升學生國際移動力：飛鷹計畫獎學金（補助出國交換學生）、雙聯學位計畫（與國外大學合作授予雙學位，約20個學程）、海外實習計畫（企業、NGO、海外華語教學實習）、國際志工服務（寒暑假至東南亞、非洲等國服務）、海外短期交流（暑期學校與語言文化營）。師大每年出國學生約300人，目標逐年增加。"
    ))
    records.append(qa(
        "師大在特殊教育方面的社會服務有哪些？",
        "師大特殊教育中心提供特殊教育鑑定、輔導與家長諮詢服務，每年服務數千名特教學生與家長。系所教師參與教育部特殊教育政策制定與課程綱要編寫。校內設有特殊教育資源教室，支援身心障礙學生學習。師大適應體育領域為身心障礙者設計運動課程，舉辦適應體育運動會。特教系學生定期至特教學校與機構實習服務。"
    ))
    records.append(qa(
        "師大的校園災害防救機制？",
        "師大設有校園災害防救應變組織，由校長擔任指揮官。定期進行地震、火災等災害演練（每學期至少一次），宿舍安全檢查與逃生演練。校園設有AED（自動體外心臟去顫器）多處，師生參加CPR+AED訓練。防災地圖與避難路線公告於各建築物。臺北市大安區為師大防災協作夥伴，提供資源與協助。"
    ))
    records.append(qa(
        "師大有哪些女性領導的系所或單位？",
        "師大多個系所與單位由女性領導：例如教育心理與輔導學系、特殊教育學系、臺灣語文學系、設計學系、幼兒與家庭科學學系等系主任為女性。師大女性教職員比例約佔全校一半以上，在師範體系中女性參與度向來較高。師大性別平等委員會亦由女教授擔任主委，確保性平政策落實。"
    ))
    records.append(qa(
        "師大如何慶祝原住民族日？",
        "師大每年8月1日（原住民族日）前後舉辦系列活動：原住民族學生成果展（手工藝、樂舞表演）、原民文化講座（邀請部落領袖與學者）、傳統美食市集。師大原住民族學生資源服務（原資中心）提供原民生課業輔導與文化支持。原資中心亦定期舉辦部落參訪，讓師生認識原住民族生活與文化。"
    ))
    records.append(qa(
        "師大永續發展推動委員會的任務是什麼？",
        "師大永續發展推動委員會成立於2022年，負責統籌校園永續發展政策與行動方案。任務包括：溫室氣體盤查與碳中和路徑規劃、校園能源效率改善（LED、太陽能）、綠色採購與循環經濟、永續教育融入課程與研究、SDGs實踐成果報告（THE Impact Rankings與STARS評比）、校園生態保育與綠建築推廣。委員會定期向校務會議報告進度。"
    ))
    records.append(qa(
        "師大資訊工程學系的研究重點有哪些？",
        "師大資訊工程學系研究涵蓋：人工智慧與機器學習（自然語言處理、電腦視覺）、軟體工程、資料科學與大數據、網路安全、人機互動、嵌入式系統。系所設有AI跨域應用研究中心，與跨域科技產業創新研究學院合作培養AI人才。師大資工系結合教育特色，在AI教育科技、智慧學習環境等跨域研究有獨特優勢。"
    ))
    records.append(qa(
        "師大在運動科學研究方面有什麼成果？",
        "師大運動科學研究涵蓋：運動生理學（運動對健康與疾病預防的影響）、運動心理學（運動員心理輔導與表現增強）、運動生物力學（動作分析與運動傷害預防）、適應體育（身心障礙者運動方案設計）。師大設有運動科學實驗室（VO2max測量、等速肌力測試等設備），研究成果發表於國際運動科學期刊，並應用於國家級運動員訓練。"
    ))
    records.append(qa(
        "師大的國際化排名與策略是什麼？",
        "師大在國際化方面表現突出：THE Impact Rankings（永續發展）名列全球前300；QS世界大學排名約700+。國際化策略包括：拓展雙聯學位與交換計畫（目標每年出國學生500人）、全英語授課課程數量提升（目前300+門，目標增至500門）、國際教師招聘與國外學者訪問、強化華語教育全球布局（MTC海外分校）、加入國際大學聯盟（如INTEI師培網絡）。"
    ))
    records.append(qa(
        "How does NTNU's Mandarin Training Center (MTC) compare globally?",
        "NTNU's Mandarin Training Center (MTC), established in 1956, is one of the oldest and largest Chinese language centers in the world. It enrolls over 4,000 students annually from more than 100 countries. MTC offers year-round programs from beginner to advanced levels, including intensive courses, business Chinese, and teacher training. Its faculty has developed widely-used textbooks like 'A Course in Contemporary Chinese' (當代中文課程). MTC also administers TOCFL (Test of Chinese as a Foreign Language), Taiwan's official Chinese proficiency test, equivalent to HSK."
    ))
    records.append(qa(
        "師大的心理與教育測驗研究發展中心是做什麼的？",
        "心理與教育測驗研究發展中心（心測中心）為師大重要的研究與服務單位，負責研發與推廣各類教育測驗與心理測量工具。最知名的成果為「國中教育會考」（國中會考）的試題研發與成績評量制度。心測中心亦開發智力測驗、性向測驗、華語文能力測驗等標準化工具，影響臺灣教育評量體系至深。"
    ))
    records.append(qa(
        "師大特殊教育學系有什麼研究特色？",
        "師大特殊教育學系為全臺最早成立的特殊教育學系之一，研究涵蓋：身心障礙教育、資優教育、融合教育、早期療育、適應體育、輔助科技（AAC）等。系所設有特殊教育中心，提供鑑定安置輔導服務。師大特教系長期參與國內特教政策制定與特教師資培育，對臺灣特殊教育發展影響深遠。"
    ))
    records.append(qa(
        "師大華語文教學系的特色是什麼？",
        "師大華語文教學系為華語教學領域的全球領航者，結合理論與實務，培養高階華語師資。課程涵蓋語言學、第二語言習得、華語文教學法、數位華語教學等。系上與國語教學中心（MTC）密切合作，學生可實際參與教學實習。畢業生遍布全球華語教學機構，系友網絡強大。"
    ))
    records.append(qa(
        "師大歐洲文化與觀光研究所有什麼課程特色？",
        "歐洲文化與觀光研究所結合歐洲區域研究與觀光產業管理，提供跨學科課程。學生需修習歐洲文化史、歐盟研究、觀光資源規劃、文化產業管理等。法語教學中心為重要附設單位。該所定期舉辦歐洲文化講座與田野調查，安排赴歐洲姊妹校交換學習。是臺灣少數專注歐洲文化與觀光的系所。"
    ))
    records.append(qa(
        "師大管理學院AACSB認證的過程是怎樣的？",
        "師大管理學院自2017年啟動AACSB認證準備工作，歷經六年努力，包括自我評鑑報告撰寫、課程學習保證（AOL）系統建立、教師學術與專業資格（AQ/PQ）分類、教學品質持續改善機制等，最終於2023年通過認證，成為亞洲師範大學體系首間獲AACSB認證的管理學院。證實師大管院教學品質與國際化達世界水準。"
    ))
    records.append(qa(
        "師大與國立臺灣大學系統的合作模式？",
        "2015年師大與臺灣大學、臺灣科技大學共同組成「國立臺灣大學系統」（NTU System）。三校學生可跨校選課（每學期至多6學分）、使用圖書館資源、參與跨校學程與活動。系統推動跨校合作研究計畫、共同舉辦學術研討會。臺大系統為臺灣最成功的大學系統合作案例，擴大三校學生的學習資源與視野。"
    ))
    records.append(qa(
        "師大圖書館有哪些特色館藏？",
        "師大圖書館特色館藏豐富：教育與心理學藏書量居全國大學之冠；中文古籍與線裝書超過10萬冊，包括珍本宋版書；梁實秋紀念藏書（含親筆手稿與批註）；臺北高等學校時期文獻（日治時期教育史料）；音樂圖書館收藏大量樂譜、CD與音樂資料。圖書館亦建置數位典藏系統，線上開放部分館藏供公眾檢索。"
    ))
    records.append(qa(
        "師大對臺灣華語文教育的貢獻？",
        "師大是臺灣華語文教育的核心推手：國語教學中心（MTC）自1956年起成為全球華語教學先驅；華語文與科技研究中心研發數位教學平臺；華語文教學系培養高階師資；TOCFL（華語文能力測驗）為官方認證考試。師大開發的《當代中文課程》等教材在全球廣泛使用。師大在世界各地設有華語教學合作據點，拓展臺灣華語教育影響力。"
    ))
    records.append(qa(
        "師大數位校史館有哪些內容？",
        "師大數位校史館（archives.lib.ntnu.edu.tw）為線上校史資料庫，收錄豐富的校史資源：歷任校長介紹、校歌試聽（多版本）、歷史照片（日治時期臺北高校至現代）、校園建築變遷、畢業紀念冊、校史影片、重要文獻與文物等。提供一般民眾與研究者瀏覽查詢，為國內大學數位校史建置之典範。"
    ))
    records.append(qa(
        "師大的校慶活動有哪些傳統？",
        "師大校慶於每年6月舉行（配合1946年省立師範學院成立日期）。傳統活動包括：校慶大會（頒發傑出校友獎與服務獎）、園遊會（各系擺攤）、體育競賽、校友回娘家座談、學術研討會。逢五逢十的「大慶」則擴大舉辦，百年校慶（2022年）系列活動長達一年，包含校史展覽、世界校友大會與國際學術論壇。"
    ))
    records.append(qa(
        "師大有哪些知名的海外校友組織？",
        "師大海內外校友組織網絡發達，海外校友會遍布美、加、日、韓、東南亞（馬來西亞、泰國、印尼）、歐洲（英國、德國）、澳洲等地。這些校友會定期舉辦聚會與學術活動，協助在校生出國留學或實習。師大校友總會與各地分會保持密切聯繫，每年校慶期間舉辦校友聯誼活動。海外校友也是師大國際招生與合作的重要橋梁。"
    ))
    records.append(qa(
        "What are some notable research achievements at NTNU?",
        "NTNU has made significant research contributions in several fields: (1) **Educational measurement** — the Psychological and Educational Testing Center develops Taiwan's national high school exams; (2) **Chinese language technology** — leading AI-driven Chinese language learning platforms; (3) **Adaptive physical education** — Asia's leading research on sports for people with disabilities; (4) **Gravity wave research** — the Astronomy and Gravity Research Center collaborates on international physics projects; (5) **Environmental education** — NTNU scores highly on STARS sustainability ratings."
    ))
    records.append(qa(
        "師大的性別平等教育和愛洛生活節是什麼？",
        "師大性別平等教育由性別平等教育委員會主導，每年舉辦「愛洛生活節」（Airo Festival）為期一個月，透過講座、影展、音樂會與展覽推廣性別平等與多元文化。議題涵蓋性別刻板印象破除、LGBTQ+人權、數位性別暴力防治、月經平權（提供免費衛生棉）等。師大在性別平等教育方面為大學標竿，設有性別友善廁所與宿舍措施。"
    ))
    records.append(qa(
        "師大的人文季是什麼活動？",
        "師大人文季由文學院創辦於2001年，每年春季（3–5月）舉辦，為期約兩個月。以「人與人文的對話」為核心精神，活動包括：文學講座（邀請作家與學者）、電影放映與座談、戲劇演出、田野調查工作坊、茶席與音樂會。人文季也與大安區社區合作，開放部分活動供民眾參與，促進大學與社會的對話。"
    ))
    records.append(qa(
        "師大在藝術創作方面的國際交流？",
        "師大藝術學院與美術館積極推動國際藝術交流：與日本東京藝術大學、京都藝術大學等簽訂交換協定；師大美術館每年舉辦國際藝術家駐村與展覽；音樂學院與歐美音樂院進行學生交流與聯合音樂會。師大藝術季與設計週邀請國際藝術家來校講座與工作坊，提升師生國際視野。"
    ))
    records.append(qa(
        "師大的師資培育學院和師資培育與就業輔導處有何不同？",
        "師資培育學院與師資培育與就業輔導處為師大兩個不同單位：師資培育學院負責教育學程課程規劃與教學，包含中等學校教師教育學程與教育實習；師資培育與就業輔導處則負責教師資格考試輔導、就業媒合與教師甄試資訊提供。兩者協作形成從培育、實習、檢定到就業的完整鏈條。"
    ))
    records.append(qa(
        "師大的北投校區計劃是什麼？",
        "師大曾規劃北投校區（位於臺北市北投區），原計劃作為擴展校地使用，但該計畫因多種因素（包括地方居民意見與經費考量）未能實現。目前師大維持和平、公館、林口三個校區的長期發展藍圖，校園空間透過新建與整建持續優化，2021年啟用師大美術館即為校園空間活化範例。"
    ))
    records.append(qa(
        "師大的體育館有哪些設施？",
        "師大和平校區體育館為綜合性運動設施，包含：室內籃球場、排球場、羽球場、桌球室、體操教室、韻律教室、重量訓練室、游泳池（溫水）。公館校區與林口校區亦設有運動場與體育館。所有場館開放師生使用，部分時段與設施亦開放社區居民收費使用。運動場館由體育室管理。"
    ))
    records.append(qa(
        "師大音樂學院和全臺唯一音樂圖書館？",
        "師大音樂學院成立於2009年（為臺灣大學中少數獨立音樂學院），設有音樂學系、流行音樂產業碩士專班與亞洲流行音樂數位科技研究中心。音樂圖書館位於和平校區，為全臺灣大學中唯一獨立設置的音樂專業圖書館，收藏包括古典樂譜、音樂CD與黑膠唱片、音樂學術期刊、以及數位音樂資料庫。音樂系館設有演奏廳與錄音室。"
    ))
    records.append(qa(
        "師大有哪些重要獎學金？",
        "師大提供多種獎學金供學生申請：師大傑出學生獎學金（獎勵優秀學業表現）、飛鷹計畫獎學金（獎助出國交換學生）、師大文學院獎學金、師大管理學院優秀學生獎學金、師大原住民族學生獎助學金、師大體育獎學金（獎助運動績優學生）。此外，設有急難救助金協助經濟困難學生。各系所亦設有自籌獎學金。"
    ))
    records.append(qa(
        "師大的校園安全措施有哪些？",
        "師大校園安全措施包括：24小時校園安全監控系統、緊急電話亭（校園各處設有緊急通報按鈕）、校園腳踏車巡邏（駐衛警與保全人員）、夜間護送服務（校警陪同步行至車站或宿舍）、宿舍門禁管理（刷卡進出）。校園緊急通報專線為(02) 7749-1110及(02) 7749-1119。每學期舉辦校園安全講座與防災演練。"
    ))
    records.append(qa(
        "How does NTNU support international students?",
        "NTNU offers comprehensive support for international students through its Office of International Affairs. Services include: visa and residence permit assistance, Chinese language courses (at MTC), international student orientation (buddy program), academic counseling, health insurance guidance, and housing services. The international student association organizes cultural events and trips. NTNU also offers scholarships specifically for international students, including the Taiwan Scholarship and MOE Huayu Enrichment Scholarship."
    ))
    records.append(qa(
        "師大的2026年以後發展藍圖是什麼？",
        "第15任校長宋曜廷於2026年就任後提出「奠基、躍升、新師大」的發展藍圖：奠基—鞏固師資培育與教育研究核心優勢；躍升—推動AI與數位轉型、強化跨域研究；新師大—提升國際化與永續發展。重點政策包括深化國立臺灣大學系統合作、擴展全球華語教育版圖、推動智慧校園建設、強化產學合作與新創生態。"
    ))
    records.append(qa(
        "師大如何推動數位轉型與AI教育？",
        "師大多管齊下推動數位轉型：成立網路大學辦公室建置線上課程平臺；開設AI跨域應用學程（跨域科技產業創新研究學院）；在課程中導入人工智慧與資料科學教學；圖書館建置數位典藏與AI檢索系統；校務行政系統雲端化與智慧化。師大亦開設AI教育相關師資培訓課程，培養中小學AI教學人才。"
    ))

    # --- More Taiwan records to round out sections ---
    records.append(qa(
        "臺灣有哪些世界遺產潛力點？",
        "臺灣雖非聯合國會員，但文化部已公告18處世界遺產潛力點，包括：玉山國家公園（自然遺產）、太魯閣國家公園（自然遺產）、阿里山森林鐵路（文化遺產）、淡水紅毛城與其周邊建築群（文化遺產）、金門戰地文化（文化遺產）、澎湖玄武岩自然保留區（自然遺產）、蘭嶼聚落與自然景觀（複合遺產）、排灣族石板屋聚落（文化遺產）等。"
    ))
    records.append(qa(
        "臺灣的溫泉文化有什麼特色？",
        "臺灣位處板塊交界帶，地熱資源豐富，溫泉遍佈全臺。著名溫泉區包括：北投溫泉（白磺、青磺，酸性硫磺泉）、烏來溫泉（碳酸氫鈉泉，美人湯）、礁溪溫泉（無色無味碳酸泉，宜蘭）、關子嶺溫泉（泥漿溫泉，世界少數，臺南）、知本溫泉（鹼性碳酸泉，臺東）。臺灣人喜愛泡湯，從日治時期發展出泡湯文化，許多溫泉區結合旅館與美食。"
    ))
    records.append(qa(
        "臺灣的離島交通方式有哪些？",
        "臺灣離島交通主要依賴航空與海運：金門、馬祖、澎湖有密集航班（立榮、華信為主），航程約50–70分鐘；綠島、蘭嶼從臺東富岡漁港搭船約50分鐘–2小時，亦有小型飛機（德安航空）。臺北松山機場為主要離島航線樞紐。離島對外交通受東北季風影響，冬季航班與船班易取消。島上交通以機車為主。"
    ))
    records.append(qa(
        "臺灣的廟宇建築有哪些特色？",
        "臺灣廟宇建築融合閩南與粵東風格，特色包括：燕尾脊（屋頂兩端翹起如燕尾）、剪黏（陶瓷片剪貼成龍鳳花鳥）、交趾陶（低溫釉陶人物與神獸）、龍柱（雕龍石柱，廟門左右）、壁畫與彩繪（門神、歷史故事）。鹿港龍山寺、臺南大天后宮、北港朝天宮為經典代表。廟宇格局一般為三川殿、正殿、後殿，層層遞進。"
    ))
    records.append(qa(
        "What is the best time to visit Taiwan?",
        "The best time to visit Taiwan is autumn (October–December) when the weather is cool and dry across the island. Spring (March–May) is also pleasant with blooming flowers and comfortable temperatures. Summer (June–September) is hot and humid with frequent typhoons, though it's great for beach activities and hiking in higher mountains. Winter (January–February) is mild in the south but cool and rainy in the north, with occasional snow on high mountains like Yushan and Hehuan."
    ))
    records.append(qa(
        "臺灣的交通系統有哪些特點？",
        "臺灣交通系統發達且便利：高鐵（2007年通車）連接臺北—高雄，最快1.5小時；臺鐵環島鐵路網完整；市區以大眾運輸（捷運：北捷、高捷、桃捷、中捷）與公車為主。YouBike（公共自行車）在各大城市普及。交通一卡通（悠遊卡、一卡通）可搭乘所有大眾運輸並在商店消費。機車密度世界最高，是臺灣人最常用的個人交通工具。"
    ))
    records.append(qa(
        "臺灣的離島有哪些特殊生態？",
        "臺灣離島生態豐富多樣：澎湖玄武岩地質與珊瑚礁生態（桶盤嶼柱狀玄武岩）、金門水鳥與鱟（活化石、古寧頭戰地生態）、馬祖的藍眼淚（夜光蟲發光現象）與燕鷗保護區、綠島的珊瑚礁與海底溫泉（朝日溫泉）、蘭嶼的達悟族傳統生態智慧（飛魚文化與地下屋）。東沙環礁國家公園為南海重要珊瑚礁生態系統。"
    ))
    records.append(qa(
        "臺灣的國家公園有哪些生態保護措施？",
        "臺灣國家公園採分區管理（核心保護區、緩衝區、遊憩區），核心區嚴格禁止開發與採集。各國家公園設有生態保育研究單位，長期監測動植物族群變化。太魯閣國家公園保護臺灣黑熊與櫻花鉤吻鮭；墾丁國家公園監測珊瑚白化；玉山國家公園維護高山生態系。國家公園也推動低衝擊旅遊、環境教育與原住民族共管機制。"
    ))
    records.append(qa(
        "臺灣傳統婚禮有哪些習俗？",
        "臺灣傳統婚禮習俗融合閩南、客家與現代元素。傳統儀式包括：提親（男方家長到女方家提親）、訂婚（交換戒指、奉茶）、迎娶（禮車隊伍、撒緣粉、過火盆）、拜堂（祭祖與拜天公）、宴客（辦桌流水席或飯店宴會）。現代婚禮簡化許多傳統，但保留文定儀式與婚宴。喜餅（中式大餅或西式禮盒）、聘金與嫁妝仍為重要環節。"
    ))
    records.append(qa(
        "臺灣有哪些重要的民間信仰節慶？",
        "臺灣重要民間信仰節慶包括：大甲媽祖遶境（農曆三月，全臺最大宗教活動）、北港朝天宮媽祖誕辰（農曆三月廿三）、保生大帝誕辰（農曆三月十五）、王船祭（東港迎王平安祭典，三年一次）、頭城搶孤（農曆七月，宜蘭）、鹽水蜂炮（元宵節，臺南）、炸寒單爺（元宵節，臺東）。每個節慶都充滿地方色彩與熱鬧氣氛。"
    ))
    records.append(qa(
        "臺灣的傳統「辦桌」文化是怎樣的？",
        "辦桌（流水席）為臺灣傳統宴客形式，常見於婚喪喜慶、廟會、選舉等場合。辦桌的特色是露天搭棚、圓桌擺設、總鋪師現場烹調，菜色豐富（通常12道以上）。傳統辦桌菜包括：魚翅羹、佛跳牆、炸湯圓、紅蟳米糕、清蒸魚、封肉等。知名總鋪師如林添盛、阿燦師（林明燦）為辦桌文化代表人物。現代婚宴逐漸轉向飯店宴會廳，但鄉村廟會仍保留辦桌傳統。"
    ))
    records.append(qa(
        "臺灣有哪些特色咖啡文化？",
        "臺灣咖啡文化蓬勃發展：古坑咖啡（雲林，臺灣咖啡發源地）、阿里山咖啡（高海拔精緻咖啡）、臺東咖啡（有機栽培）。連鎖咖啡店（星巴克、路易莎、Cama）遍布全臺，獨立咖啡館密度極高（臺北大安區、臺中草悟道、高雄鹽埕區為熱區）。超商咖啡（7-ELEVEN CITY CAFE、全家Let's Café）為全臺最大咖啡通路，年銷數億杯。咖啡已是臺灣人日常飲品。"
    ))
    records.append(qa(
        "臺灣的蘭花產業有多發達？",
        "臺灣為全球蘭花王國，蝴蝶蘭（Phalaenopsis）出口量世界第一，佔全球約三分之一市場。臺南後壁的「臺灣蘭花生物科技園區」為全球最大蘭花產區。蝴蝶蘭因育種技術先進（每年推出新品種）、品質穩定、運輸適應力強，外銷美國、日本、歐洲、東南亞等市場。臺灣國際蘭展（TIOS）為全球三大蘭展之一。蘭花產業年產值超過百億新臺幣。"
    ))
    records.append(qa(
        "臺灣常見的野生動物有哪些？",
        "臺灣常見野生動物包括：臺灣獼猴（全臺低海拔山區常見）、松鼠（赤腹松鼠、條紋松鼠）、白鼻心（果子狸）、鼬獾、野豬、山羌（臺灣最小鹿科動物）、穿山甲（保育類）。鳥類方面，臺灣藍鵲（臺灣特有種）、五色鳥、臺灣紫嘯鶇、黑面琵鷺（曾文溪口，國際級候鳥）。臺灣黑熊為最大型陸生哺乳動物，野外數量約200–600隻。"
    ))
    records.append(qa(
        "臺灣有哪些特有種鳥類？",
        "臺灣特有種鳥類數量豐富（約25種），著名者包括：臺灣藍鵲（Urocissa caerulea，紅喙藍羽，為臺灣最美鳥類之一）、臺灣帝雉（Syrmaticus mikado，黑長尾雉，千元鈔票上的鳥）、臺灣紫嘯鶇（Myophonus insularis）、臺灣山鷓鴣（Arborophila crudigularis）、烏頭翁（Pycnonotus taivanus）、黃山雀（Machlolophus holsti）、火冠戴菊鳥（Regulus goodfellowi，高海拔）。阿里山、大雪山、太魯閣為賞鳥熱區。"
    ))
    records.append(qa(
        "臺灣的登山文化有多盛行？",
        "臺灣登山風氣盛行，百岳（海拔3,000m以上高山100座）為許多登山者的目標。玉山主峰每年超過5萬人次申請攀登。熱門登山路線包括：玉山主峰線、雪山主東線、嘉明湖路線、能高安東軍縱走、奇萊主北峰等。國家公園管理處實施入園與入山申請制度。登山教育與裝備要求日趨專業。2020年山林開放政策後參與人數大幅成長。高山協作（布農族為主）為登山產業重要角色。"
    ))
    records.append(qa(
        "師大天文與重力研究中心在做什麼研究？",
        "師大天文與重力研究中心（ARG）隸屬理學院，從事天文物理與重力波相關前沿研究。研究團隊參與國際重力波觀測（LIGO/Virgo合作），分析重力波訊號以探索黑洞與中子星合併事件。中心亦進行宇宙學模擬、星系演化、暗物質等理論研究。ARG配備高效能運算設備，與中研院天文所及國際天文臺保持合作。"
    ))
    records.append(qa(
        "師大在媒體與傳播方面有哪些學系或課程？",
        "師大並未設立獨立的新聞或大眾傳播學系，但相關課程分散於：圖文傳播學系（科技與工程學院，培養影視、攝影、數位媒體與傳播科技人才）、國文學系（新聞寫作與編輯相關課程）、東亞學系（政治傳播與國際關係）。師大亦設有媒體素養與假訊息辨識相關通識課程。學生活動方面，師大學生會與各式校園媒體（師大新聞、師青報）提供實務經驗。"
    ))
    records.append(qa(
        "師大的學校象徵物有哪些？",
        "師大代表象徵包括：木鐸（校徽核心，象徵教育警世覺民）、師大紅與師大藍（代表色，紅：熱情活力，藍：穩重理性）、阿勃勒（校樹，黃金雨，5–6月開花）、紫荊花（校花，春季開花）。吉祥物方面，師大運動代表隊以「師大藍鵲」為形象（2017年設計），結合臺灣藍鵲與師大識別元素。這些象徵物廣泛應用於校園景觀、紀念品與文宣。"
    ))
    records.append(qa(
        "師大圖書館公館分館有什麼特色？",
        "師大圖書館公館分館位於公館校區，於1991年完工啟用，主要服務理學院、科技與工程學院、國際與社會科學學院師生。館藏以理工、科技、社會科學圖書與期刊為主。設有自習室、討論室與多媒體區。公館分館緊鄰臺大與臺科大，三校師生可透過臺大系統跨校借書。公館校區學生可在此完成大部分學習與研究需求。"
    ))
    records.append(qa(
        "師大目前有哪些線上課程和MOOC？",
        "師大在各大MOOC平臺（ewant、Coursera、FutureLearn等）開設多門線上課程，涵蓋華語教學、教育心理、人工智慧、臺灣文化等領域。國語教學中心開發的線上華語課程（如「當代中文課程」）為最受歡迎的課程。師大網路大學辦公室負責線上課程的開發與管理。2023年起師大開放部分課程為免費線上旁聽（openNTNU），推廣開放教育資源。"
    ))
    records.append(qa(
        "What courses can you take at NTNU in English?",
        "NTNU offers over 300 English-taught courses across disciplines. Full English degree programs include: Global Studies (Bachelors in International Studies), Physics (English Bachelor Program), and Education (English-taught Master's programs). Individual English courses are available in business management, computer science, psychology, Chinese language pedagogy, East Asian studies, and more. NTNU also participates in EMI (English as a Medium of Instruction) programs supported by Taiwan's Ministry of Education, aiming to double English-taught offerings by 2030."
    ))
    records.append(qa(
        "師大如何協助學生職涯發展？",
        "師大師資培育與就業輔導處提供完整職涯服務：就業博覽會（每年3月，邀請上百家企業設攤）職涯諮詢（一對一面談與履歷健檢）、實習媒合（國內外實習機會）、產業講座與企業參訪、校友職涯分享。師大畢業生就業率約85%，其中教育領域（教師）為最大就業方向，其次為文教產業、公務體系、科技業與服務業。"
    ))
    records.append(qa(
        "師大有哪些研究所不收學費或提供全額獎學金？",
        "師大提供多種研究生獎助方案：優秀研究生獎學金（核予每月8,000–12,000元，為期一年）、國科會博士生研究獎學金（每月40,000元）、教育部博士生獎學金（每年最高24萬元）。僑生與外國學生可申請臺灣獎學金（教育部）或華語文獎學金。部分產學合作碩博士班提供全額學費補助。師大也設有教學助理（TA）與研究助理（RA）制度，提供研究生服務獎助金。"
    ))
    records.append(qa(
        "師大學生藝術創作有哪些展演機會？",
        "師大學生藝術創作展演機會豐富：美術系年度畢業美展（於師大美術館或校外展場舉辦）、設計系畢業展（新一代設計展參展）、音樂系年度音樂會與畢業音樂會（國家音樂廳或校內演奏廳）、國文系紅樓文學獎與文學創作展、師大藝術季（跨學院聯合展演）。學生亦可申請校內展演空間舉辦個展或聯展。師大美術館提供學生策展實習機會。"
    ))
    records.append(qa(
        "師大在臺灣半導體產業的人才培育？",
        "師大雖非傳統半導體強校，但在半導體領域逐步布局：電機工程學系開設VLSI設計、半導體元件等課程；機電工程學系涉足半導體製程設備與智慧製造；跨域科技產業創新研究學院與業界合作半導體人才培育計畫。師大與臺積電、聯發科等企業開設產學合作實習課程與專題研究。物理學系與化學系在半導體材料與物理研究上有相關研究成果。"
    ))
    records.append(qa(
        "師大在推廣華語文教育方面的海外布局？",
        "師大在海外華語文教育布局廣泛：與美國、日本、韓國、越南、泰國、印尼等國大學合作設立華語教學中心或華語學程；每年派遣華語教師至海外教學（教育部華語教師外派計畫）；開發線上華語課程（全球學習者可遠端學習）；TOCFL華語文能力測驗海外考場遍布全球數十國；MTC與海外機構合作開設暑期華語營。師大堪稱臺灣華語文教育全球輸出的核心引擎。"
    ))
    records.append(qa(
        "師大在學生心理健康方面有哪些資源？",
        "師大學務處學生輔導中心（諮商中心）提供免費的心理諮商服務，包括個別諮商（每學期可預約6–8次）、團體諮商（壓力管理、人際關係、情緒調適等主題）、心理測驗、危機處理（自殺防治與緊急介入）。中心亦舉辦心理健康講座與工作坊（正念、放鬆訓練、時間管理等）。導師制度與同儕輔導員亦為學生心理支持網絡的一環。"
    ))
    records.append(qa(
        "師大體育與運動科學系的研究方向有哪些？",
        "師大體育與運動科學系研究涵蓋六大領域：運動生理學（運動訓練與體能評估）、運動生物力學（運動技術分析、運動傷害預防）、運動心理學（競賽心理、運動員輔導）、運動社會學（運動與社會互動）、運動管理學（運動產業與行銷）、適應體育（身心障礙者運動方案設計與評估）。系上設有運動科學實驗室，擁有等速肌力測量儀、VO2max分析儀等設備。"
    ))
    records.append(qa(
        "師大美術館開館以來的重點展覽有哪些？",
        "師大美術館於2021年開幕，為臺灣首座大學附屬美術館。開館以來重點展覽包括：《師大美術館開館展——匯流與奔放》（介紹師大美術百年脈絡）、《梁實秋文學展》與《師大百年校慶特展》、《臺灣前輩藝術家特展》（陳澄波、廖繼春等）、《國際藝術家交流展》（與日本、韓國等姊妹校合作）。美術館同時舉辦教育工作坊與社區互動活動，連結大學與公眾。"
    ))
    records.append(qa(
        "師大管理學院有哪些學系和研究所？",
        "師大管理學院設有：企業管理學系（學士、碩士、博士班）、管理學研究所（MBA）、國際企業管理碩士學程（IMBA，全英語授課）。此外設有智能與指數化投資研究中心（SII，研究FinTech與智能投資）與師大育成中心（管理學院負責）。管院於2023年通過AACSB認證，為亞洲師範大學首例，代表其教學品質與國際化獲全球認可。"
    ))
    records.append(qa(
        "師大在物理教育與科普推廣方面做了哪些事？",
        "師大物理學系設有物理演示廳（公館校區），定期舉辦中小學生科學營隊與物理演示活動，每年服務數千名學童。物理系教師參與教育部科普計畫，製作科普影片與教材。師大科學教育中心開發物理教育課程與評量工具。師大亦主辦全國物理競賽（物理奧林匹亞選訓），培育優秀物理人才。"
    ))
    records.append(qa(
        "師大「誠正勤樸」校訓的具體涵義是什麼？",
        "「誠正勤樸」為師大第三任校長劉真於1950年代訂定的校訓。誠：真誠無偽，對己對人皆以誠相待。正：正直不阿，追求公義與道德。勤：勤奮不懈，努力向學、敬業樂群。樸：樸實無華，不尚虛華、腳踏實地。此四字貫穿師大教育理念，也是師大學生與校友的核心品格象徵。校訓碑立於和平校區行政大樓前。"
    ))
    records.append(qa(
        "師大未來十年（2026–2036）的主要發展目標有哪些？",
        "師大2026年新任校長宋曜廷提出「奠基、躍升、新師大」藍圖：奠基包括強化師資培育、深化教育研究；躍升包括邁向AI數位校園、擴充產學合作與跨域研究；新師大包括打造國際化校園、達成校園碳中和目標。具體目標包括：國際學生比例提高至20%、開設500門全英語課程、成立AI教育研究中心、綠能校園投資等。這些目標將於2026–2036年逐步推進。"
    ))
    records.append(qa(
        "臺灣有哪些重要的現代文學作家？",
        "臺灣現代文學作家陣容豐富：白先勇（《臺北人》、《孽子》）、黃春明（鄉土文學代表，《鑼》、《兒子的大玩偶》）、陳映真（左翼批判文學）、王文興（《家變》）、七等生（獨特文風）、吳晟（鄉土詩人）、簡媜（散文家，師大國文系校友）、駱以軍（後現代小說）、吳明益（自然書寫，《複眼人》、《天橋上的魔術師》）。臺灣文學在華語世界擁有獨特地位。"
    ))
    records.append(qa(
        "臺東池上為什麼米特別有名？",
        "池上米產於臺東縣池上鄉，以高品質聞名全國。池上鄉位於花東縱谷，來自中央山脈與海岸山脈的純淨水源灌溉，日夜溫差大、土壤肥沃，適合水稻生長。池上米以「池上便當」（鐵路便當始祖）聞名，使用池上米製作的便當米飯Q彈香甜。池上米是臺灣少數取得地理標示認證的農產品（「池上米」商標）。金城武樹（伯朗大道）亦位於池上，帶動觀光。"
    ))
    records.append(qa(
        "屏東東港的迎王平安祭典是什麼？",
        "東港迎王平安祭典（東港王船祭）為屏東東港三年一度的重要宗教盛事（逢丑、辰、未、戌年舉行）。祭典由東港東隆宮主辦，主要儀式包括：請王（迎請千歲爺）、過火、王船繞境、送王（火燒王船）。王船為木造或紙糊船，搭載千歲爺離境。為臺灣保存最完整的王船信仰文化，被文化部指定為國家重要民俗。"
    ))
    records.append(qa(
        "臺灣的殯葬文化中「孝女白琴」是什麼？",
        "「孝女白琴」（又稱孝女白瓊）為臺灣喪葬文化中的特殊職業，起源於1970年代。通常由女性扮演，穿著白色孝服、手持麥克風，在喪禮現場以哭調（哭腔）演唱哀歌，表達孝思。表演內容包括哭爹哭娘、哭人生無常等。孝女白琴在現代逐漸轉型或減少，但在鄉村地區仍可見到。此現象反映臺灣喪禮中的戲劇化與表演性特色。"
    ))
    records.append(qa(
        "臺灣的古蹟保存運動有哪些重要案例？",
        "臺灣古蹟保存運動重要案例包括：臺北迪化街保存運動（1980年代，保留大稻埕清代老街區建築）、高雄哈瑪星保存（日治時期現代化港區）、臺中彩虹眷村（爭取保留老舊眷村並彩繪改造）。鹿港、淡水、大溪、安平等歷史街區亦透過社區營造進行保存。文化資產保存法（1982年制定，多次修訂）為古蹟保存的主要法律依據。民間團體（如臺灣歷史資源經營學會）在保存運動中扮演關鍵角色。"
    ))
    records.append(qa(
        "臺灣有哪些知名的流行音樂歌手？",
        "臺灣流行音樂在全球華語樂壇居領導地位。經典歌手：鄧麗君（《月亮代表我的心》、《但願人長久》）、周杰倫（《青花瓷》、《七里香》）、蔡依林（《舞孃》、《日不落》）、五月天（樂團，《倔強》、《溫柔》）、張惠妹（《聽海》、《三天三夜》）、林俊傑（新加坡但在臺灣發展，《江南》、《修煉愛情》）。新世代歌手：周興哲、鄧紫棋（香港但在臺發展）、告五人、茄子蛋（樂團）。金曲獎（Golden Melody Awards）為華語流行音樂最高榮譽。"
    ))
    records.append(qa(
        "臺灣各地有哪些代表性的傳統市場？",
        "臺灣各城市代表性傳統市場包括：臺北南門市場（南北貨與熟食）、大稻埕永樂市場（布料與食物）、東三水街市場（萬華）；臺中新光黃昏市場（大型黃昏市集）；臺南水仙宮市場（府城老市場）、東菜市（百年市場）；高雄鹽埕第一市場（在地小吃）；宜蘭市場。傳統市場與夜市同樣是臺灣庶民生活文化的核心，近年推動改造計劃（如臺北東三水街成功轉型）。"
    ))
    records.append(qa(
        "臺灣有哪些知名的國際級自行車賽事？",
        "臺灣國際知名自行車賽事包括：臺灣自行車登山王挑戰（KOM，從花蓮七星潭到武嶺，爬升3,275公尺，被全球自行車媒體評為世界最困難挑戰路線之一）、環臺賽（Tour de Taiwan，UCI 2.1級，每年3月舉行，途經各縣市）、NeverStop永不放棄系列賽（長距離自我挑戰）。臺灣優質的自行車基礎設施與多變地形，使其成為國際自行車旅遊熱點。"
    ))
    records.append(qa(
        "臺灣人為什麼喜歡存股？投資文化？",
        "臺灣散戶（個人投資者）參與股市比例極高（約占交易量60%以上），存股文化特別盛行。原因包括：定存利率長期偏低（1%以下）、高股息ETF（如0050、0056、00878）普及、政府推動「健全股市措施」吸引散戶。臺灣每年現金股利殖利率約3–5%，在全球名列前茅。年輕人透過定期定額投資ETF存股蔚為風潮。臺灣股市總開戶數約1,200萬戶，顯示全民投資參與度之高。"
    ))
    records.append(qa(
        "臺灣有哪些重要的民間文學與傳說？",
        "臺灣民間文學與傳說豐富：林投姐（被負心漢拋棄的女鬼復仇故事）、虎姑婆（警告兒童勿輕信陌生人的童話）、白賊七（騙子故事，類似華南地區的機智人物）、廖添丁（日治時期義賊，劫富濟貧的傳奇人物）、基隆雨神（關於雨都的神話）。這些民間故事反映臺灣社會歷史與庶民價值觀。近年文化部推動民間文學採集與出版，保存口傳文學。"
    ))
    records.append(qa(
        "臺灣的百貨公司與購物中心有哪些特色？",
        "臺灣百貨密度極高（全球排名前茅），主要集團有新光三越、遠東Sogo、微風集團、統一時代。臺北信義區為百貨一級戰區（新光三越A4/A8/A9/A11、微風南山、Bellavita、101購物中心）。購物中心方面：臺中三井Outlet、高雄義大世界、桃園華泰名品城（Outlet）。日本品牌（UNIQLO、MUJI、唐吉訶德、壽司郎等）大規模進駐。臺灣百貨以餐飲占比高（約30–40%）為特色。"
    ))
    records.append(qa(
        "臺灣的民宿文化是怎樣的？",
        "臺灣民宿蓬勃發展，為觀光重要特色。全臺合法民宿超過1萬家，以宜蘭、花蓮、南投、澎湖最密集。民宿類型多樣：山景民宿（清境、阿里山）、海景民宿（花東海岸、澎湖）、農村民宿（宜蘭、苗栗）、老宅改建民宿（臺南、大溪）、特色主題民宿（童話、工業風、寵物友善）。民宿管理辦法（2001年制定）規範設立條件與經營方式。臺灣民宿以人情味與特色風格著稱，與飯店形成差異化。"
    ))
    records.append(qa(
        "臺灣的棒球文化有多深厚？",
        "棒球為臺灣最具代表性的運動，被稱為「國球」。日治時期（1895年後）日本引入棒球，全臺各級學校推廣。1969年金龍少棒隊贏得威廉波特世界少棒錦標賽冠軍，掀起全臺棒球熱潮。職棒（中華職棒CPBL）成立於1990年，現有6支球隊（兄弟象、統一獅、樂天桃猿、味全龍、富邦悍將、臺鋼雄鷹）。臺灣球員（王建民、陳偉殷、陽岱鋼、林子偉、張育成）活躍於美國職棒MLB。棒球場遍佈全臺各縣市。"
    ))
    records.append(qa(
        "師大在臺灣電影與紀錄片領域的影響？",
        "師大雖無電影學系，但相關課程與活動豐富：圖文傳播學系開設電影製作、紀錄片與攝影課程；國文系開設電影文學與劇本寫作；師大學生會與各社團定期舉辦電影節與紀錄片放映座談。著名紀錄片導演楊力州（師大美術系／設計研究所校友）作品《拔一條河》、《紅毯》等享譽國際。師大國際事務處與藝術學院定期舉辦國際學生短片競賽。"
    ))
    records.append(qa(
        "師大學生在課外活動方面的時間分配如何？",
        "師大學生課外活動參與度高，常見組合為：課業與社團並重（系學會、服務性社團、體能性社團）、部分學生參與打工或家教、部分投入競技運動（校隊練習每週數次）。師大社團約174個，近半數學生參與至少一個社團。畢業門檻要求學生參與服務學習（志工服務至少18小時）。師大學生學業壓力中等，校風自由，鼓勵學生探索多元興趣。"
    ))
    records.append(qa(
        "師大有什麼餐飲或小吃的獨特校園文化？",
        "師大因師生來自全臺與各國，校園餐飲文化多元。師大夜市商圈聞名遐邇，學生最常造訪的小吃包括：師大鹽水雞（和平校區側門）、許記生煎包、燈籠滷味、馬來西亞咖哩雞、夜市口的車輪餅。師大校內地下餐廳（地餐）以平價自助餐與簡餐為主力。公館校區則有公館商圈多樣選擇。師大周邊為臺北著名美食區之一，吸引外地遊客專程造訪。"
    ))
    records.append(qa(
        "師大在臺灣教育史上最重要的貢獻是什麼？",
        "師大在臺灣教育史最重要的貢獻為：作為師資培育的「教育總引擎」。從1946年省立師範學院迄今，師大培育數十萬名中學教師，影響全臺數代學子。師大主導臺灣課程改革（九年一貫、十二年國教）、教育測驗研發（國中會考）、教育政策智庫（教育研究集刊）。在華語文教育方面，MTC為全球華語教學領航者。簡言之，師大不僅是教育工作者的大本營，更是臺灣教育現代化與國際化的核心推手。"
    ))
    records.append(qa(
        "How would you describe NTNU's overall campus culture?",
        "NTNU's campus culture balances academic rigor with creative vibrancy. As a former teacher's college now transformed into a comprehensive university, it maintains a strong sense of social responsibility while embracing innovation. The Heping campus blends historic Japanese-era red-brick buildings (city-designated heritage) with modern facilities. Students describe the atmosphere as 'free and open' — the school motto of 誠正勤樸 (Integrity, Righteousness, Diligence, Simplicity) still resonates. Music performances, art exhibitions, and sports events fill the calendar year-round. The university's location in Da'an District — Taipei's cultural and food heartland — adds to its lively character."
    ))
    records.append(qa(
        "師大學生畢業後的薪資狀況如何？",
        "師大畢業生平均薪資依科系而異：教育相關科系（教師）起薪約新臺幣38,000–45,000元（含導師費與輔導費）；資訊、電機類科系（工程師）起薪約45,000–55,000元；管理類科系起薪約35,000–42,000元。師大畢業生就業穩定度高（尤其教師與公務人員），長期薪資成長平穩。勞動部統計師大畢業生畢業後五年平均薪資約50,000–60,000元。"
    ))
    records.append(qa(
        "師大在泰國設有什麼合作機構？",
        "師大在泰國的合作與影響：與泰國朱拉隆功大學、法政大學等簽訂交換協定；MTC每年開設泰國華語教師培訓班；師大華語文教學系與泰國教育部合作推廣華語教育；師大在泰國華語文能力測驗（TOCFL）考場服務數千名考生。泰國為師大境外招生重點國家，每年約有30–50名泰國學生在師大就讀學位或語言課程。"
    ))
    records.append(qa(
        "臺灣的夜市有哪些是24小時或營業到很晚的？",
        "臺灣多數夜市營業時間約為傍晚6點至凌晨12點，但部分夜市營業至深夜：臺北士林夜市部分攤位營業至凌晨1–2點；臺中逢甲夜市攤位大多營業至凌晨1點；高雄六合夜市以觀光客為主，營業至午夜。24小時營業的餐飲選項以超商（7-ELEVEN、全家）、豆漿店（如四海遊龍、永和豆漿）與部分小吃店為主，而非整個夜市。"
    ))
    records.append(qa(
        "臺灣的醫療旅遊發展情況如何？",
        "臺灣醫療旅遊以健檢、醫美、眼科（近視雷射）、牙科、人工生殖為主要項目。國際醫療服務中心（臺大、榮總、長庚等醫學中心設有）提供一站式服務。主要客源來自中國大陸（2019年前）、東南亞（越南、印尼、菲律賓）、日本與美國。臺灣醫療的優勢在於高品質（JCI國際醫院評鑑多家通過）、相對低廉費用（約為美國的1/5–1/3）與先進技術。政府推動醫療旅遊專區與國際醫療行銷。"
    ))
    records.append(qa(
        "淡水紅毛城的歷史演變？",
        "淡水紅毛城位於淡水河口，最早由西班牙人於1629年興建（聖多明哥城），1642年荷蘭人驅逐西班牙人後改建。名稱中的「紅毛」為漢人對荷蘭人的俗稱。清領時期英國於1867年租借紅毛城作為領事館，增建英國領事官邸（紅磚洋樓）。1980年英國撤館後由中華民國政府接收。現為臺北市定國定古蹟，為臺灣最具代表性的西洋建築之一。"
    ))
    records.append(qa(
        "臺北捷運的歷史和規模如何？",
        "臺北捷運（Metro Taipei）於1996年3月28日木柵線通車，目前營運路線包括：文湖線、淡水信義線、松山新店線、中和新蘆線、板南線、環狀線（第一階段），共約131站、營運里程約152公里。日均運量約200萬人次。臺北捷運以整潔安全聞名，禁食文化獨樹一格（站內禁止飲食）。悠遊卡（EasyCard）為主要支付工具，亦可搭乘公車、YouBike與臺鐵。"
    ))
    records.append(qa(
        "臺灣原住民族有什麼重要的手工藝品？",
        "臺灣原住民族手工藝豐富多元：阿美族—陶壺、編織（月桃葉、苧麻）、木雕；泰雅族—織布（菱形紋象徵祖靈眼睛）、口簧琴製作；排灣族—琉璃珠（蜻蜓珠）、青銅刀、刺繡；布農族—百步蛇圖騰木雕、苧麻編織；魯凱族—石板雕刻、百合花圖騰刺繡；達悟族—拼板舟（精細木工）、銀盔、丁字褲紡織。這些傳統工藝展現各族的宇宙觀與社會組織。"
    ))
    records.append(qa(
        "臺灣有哪些結合在地文化的特色旅館？",
        "臺灣特色旅館與在地文化結合緊密：臺北北投麗禧溫泉酒店（溫泉文化）、南投日月潭涵碧樓（湖景與邵族文化）、臺南天下南隅（府城歷史風格）、花蓮理想大地（西班牙式建築結合縱谷景觀）、嘉義桃城茶樣子（阿里山茶文化主題）、宜蘭蘭城晶英（紅磚建築合院風格）。老屋改建民宿（如臺南、大溪、鹿港古宅民宿）亦為熱門選擇。文化部推動的「臺灣旅館文創化」鼓勵旅館融入在地元素。"
    ))
    records.append(qa(
        "臺灣有哪些重要的現代建築地標？",
        "臺灣現代建築地標包括：臺北101（2004年完工，高509m，曾為世界最高樓，象徵臺灣經濟實力）、臺中國家歌劇院（伊東豊雄設計，世界首創曲牆結構）、高雄國家體育場（世運主場館，龍騰造型，太陽能發電屋頂）、臺北表演藝術中心（士林，OMA設計，球體劇場懸浮造型）、衛武營國家藝術文化中心（高雄，荷蘭Mecanoo設計）等。這些建築展現臺灣當代設計與工程實力。"
    ))
    records.append(qa(
        "師大和平校區的歷史建築有哪些故事？",
        "師大和平校區保留了日治時期臺北高等學校的多棟建築：行政大樓（1929年，哥德復興風格紅磚建築，原為高校本館，見證百年教育史）；文薈廳（原高校禮堂，木造結構，梁柱以榫接方式固定，經常有音樂會與展演）；普字樓（原高校普通教室）。這些建築於2008年指定為臺北市定古蹟（「原臺灣師範大學高等學校校舍」），紅磚拱圈與尖拱窗為經典特色。"
    ))
    records.append(qa(
        "師大在學術倫理與研究誠信方面有什麼措施？",
        "師大設有研究誠信辦公室與學術倫理委員會，負責推動學術倫理教育與審查案件。所有研究生必修學術倫理課程（線上課程6小時或實體講座）。教師與研究人員執行研究計畫前需完成學術倫理訓練。本校遵守科技部（國科會）研究誠信規範，每年舉辦學術倫理研討會。違反學術倫理（如抄襲、捏造數據）者依情節處以撤銷學位、停權等處分。師大在學術倫理教育方面為全國大學標竿之一。"
    ))
    records.append(qa(
        "師大有學生經營的咖啡廳或商店嗎？",
        "師大設有學生實習商店與學生社團經營的空間：師大學生會經營的福利社（販售文創商品與點心）；管理學院學生開設的實習咖啡館（由企業管理學系經營，提供實務經營經驗）；圖書館內的學生書店（師大出版中心學生門市）。這些空間由學生自主管理、營運與行銷，作為創新創業的試驗場域。師大育成中心也協助學生團隊申請微型創業補助。"
    ))
    records.append(qa(
        "師大校園內有哪些公共藝術作品？",
        "師大校園內公共藝術作品豐富：太極銅質雕像（2017年北京大學贈送，象徵兩校友誼與太極文化）；自由之鐘（和平校區，象徵學術自由與開放精神）；各建築物內外設置的當代藝術作品（雕塑、壁畫、裝置藝術）；美術館前廣場的戶外雕塑。師大校園本身就是一座開放式美術館，藝術作品與百年歷史建築相互輝映，使校園充滿人文氣息。"
    ))
    records.append(qa(
        "師大在運動傷害防護與運動醫學方面的資源？",
        "師大運動傷害防護資源由體育與運動科學系與運動競技學系共同提供：運動傷害防護室（提供學生運動員即時傷病處理與復健指導）；運動科學實驗室（生物力學分析與功能性評估）。校隊選手配有運動防護員。師大與臺大醫院、三軍總醫院等合作運動醫學門診。師大每年舉辦運動傷害防護研習會，提供師生與教練專業訓練。"
    ))
    records.append(qa(
        "師大物理學系和臺大物理系有什麼合作？",
        "師大物理學系與臺大物理系合作密切：兩校同屬國立臺灣大學系統，學生可跨校修課與使用實驗室；共同舉辦物理研討會與學術講座；部分教授合聘（兩校共同聘任）；在理論物理與高能物理領域有聯合研究計畫。師大天文與重力研究中心與臺大物理系亦有合作。此合作模式讓師大物理系學生可接觸臺大資源，拓展學習視野。"
    ))

    return records
