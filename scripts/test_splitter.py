
import logging
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

# Configure logging to see Haystack's internal processing info if needed
logging.basicConfig(level=logging.INFO)

def test_splitter_behavior():
    # 1. Simulate the data exactly as seen in the issue
    # This is the raw text constructed in load_csv_as_documents
    raw_text = (
        "标题: 注射用乳糖酸红霉素\n"
        "编号: 国药准字H43020028\n"
        "药品性质: 处方药\n"
        "性状: 白色或类白色的结晶或粉末或疏松块状物。\n"
        "适应症: 本品作为青霉素过敏患者治疗下列感染的替代用药： 1.溶血性链球菌?肺炎链球菌等所致的急性扁桃体炎?急性咽炎?鼻窦炎。 2.溶血性链球菌所致的猩红热?蜂窝织炎?白喉及白喉带菌者?气性坏疽?炭疽?破伤风?放线菌病?梅毒?单核细胞增多性李斯特菌病等。 3.军团菌病，支原体肺炎，衣原体肺炎，其他衣原体属?支原体属所致泌尿生殖系感染。 4.沙眼衣原体结膜炎，厌氧菌所致口腔感染，空肠弯曲菌肠炎，百日咳。\n"
        "不良反应: 1.胃肠道反应多见，有腹泻?恶心?呕吐?中上腹痛?口舌疼痛?胃纳减退等，其发生率与剂量大小有关。\n"
        "2.肝毒性少见，患者可有乏力?恶心?呕吐?腹痛?发热及肝功能异常，偶见黄疸等。\n"
        "3.大剂量(≥4g/日)应用时，尤其肝?肾疾病患者或老年患者，可能引起听力减退，主要与血药浓度过高(>12mg/L)有关，停药后大多可恢复。\n"
        "4.过敏反应表现为药物热?皮疹?嗜酸粒细胞增多等，发生率约0.5%～1%。\n"
        "5.其他：偶有心律失常?口腔或阴道念珠菌感染。\n"
        "用法用量: 1.静脉滴注：成人一次0.5～1.0g，一日2～3次，治疗军团菌病剂量需增加至一日3～4g，分4次滴注，小儿每日按体重20～30mg/kg，分2～3次滴注。2.乳糖酸红霉素滴注液的配制： (1)先加灭菌注射用水10ml至0.5g乳糖酸红霉素粉针瓶中或加20ml至1g乳糖酸红霉素粉针瓶中，用力震摇至溶解。 (2)然后加入生理盐水或其它电解质溶液中稀释，缓慢静脉滴注，注意红霉素浓度在1%～5%以内。 (3)溶解后也可加入含葡萄糖的溶液稀释，但因葡萄糖溶液偏酸性，必须每100ml溶液中加入4%碳酸氢钠1ml。\n"
        "禁忌: 对红霉素类药物过敏者禁用。\n"
        "注意事项: 1.溶血性链球菌感染用本品治疗时，至少需持续10日，以防止急性风湿热的发生。\n"
        "2.肾功能减退患者一般无需减少用量。\n"
        "3.用药期间定期随访肝功能。肝病患者和严重肾功能损害者红霉素的剂量应适当减少。\n"
        "4.患者对一种红霉素制剂过敏或不能耐受时，对其他红霉素制剂也可过敏或不能耐受。\n"
        "5.对诊断的干扰：本品可干扰Higerty法的荧光测定，使尿儿茶酚胺的测定值出现假性增高，血清碱性磷酸酶?胆红素?丙氨酸氨基转移酶和门冬氨酸氨基转移酶的测定值均可能增高。\n"
        "6.因不同细菌对红霉素的敏感性存在一定差异，故应做药敏测定。\n"
        "孕妇及哺乳期妇女用药: 1.本品可通过胎盘而进入胎儿循环，浓度一般不高，文献中也无对胎儿影响方面的报道，但孕妇应用时仍宜权衡利弊。 2.本品有相当量进入母乳中，哺乳期妇女应用时应暂停哺乳。\n"
        "儿童用药: 尚不明确\n"
        "药物相互作用: 1.本品可抑制卡马西平和丙戊酸等抗癫痫药的代谢，导致其血药浓度增高而发生毒性反应。\n"
        "2.本品与阿芬太尼合用可抑制后者的代谢，延长其作用时间。\n"
        "3.本品与阿司咪唑或特非那定等抗组胺药合用可增加心脏毒性，与环孢素合用可使后者血药加而产生肾毒性。\n"
        "4.本品与氯霉素和林可酰胺类有拮抗作用，不推荐合用。\n"
        "5.本品为抑菌剂，可干扰青霉素的杀菌效能，故当需要快速杀菌作用如治疗脑膜炎时，两者不宜同时使用。\n"
        "6.长期服用华法林的患者应用本品时可导致凝血酶原时间延长，从而增加出血的危险性，老年病人尤应注意。两者必须同时使用时，华法林的剂量宜适当调整，并严密观察凝血酶原时间。\n"
        "7.除二羟丙茶碱外，本品与黄嘌呤类合用可使氨茶碱的肝清除减少，导致血清氨茶碱浓度升高和(或)毒性反应增加，这一现象在合用6日后较易发生，氨茶碱清除的减少幅度与本品血清峰值成正比，因此在两者合用疗程中和疗程后，黄嘌呤类的剂量应予调整。\n"
        "8.本品与其他肝毒性药物合用可能增强肝毒性。\n"
        "9.本品与耳毒性药物合用，尤其肾功能减退患者可能增加耳毒性。\n"
        "9.本品与洛伐他丁合用时可抑制其代谢而使血浓度上升，可能引起横纹肌溶解。\n"
        "10.与咪达唑仑或三唑仑合用时可减少两者的清除而增强其作用。\n"
        "药理毒理: 1.本品属大环内酯类抗生素，为水溶性的红霉素乳糖醛酸酯，对葡萄球菌属、各组链球菌和革兰阳性杆菌均具抗菌活性，奈瑟菌属、流感嗜血杆菌、百日咳鲍特氏菌等也可对本品呈现敏感，对除脆弱拟杆菌和梭杆菌属以外的各种厌氧菌亦具抗菌活性。\n"
        "2.本品对军团菌属、胎儿弯曲菌、某些螺旋体、肺炎支原体、立克次体属和衣原体属有抑制作用。\n"
        "3.本品系抑菌剂，但在高浓度时对某些细菌也具杀菌作用。\n"
        "4.本品可透过细菌细胞膜，在接近供位（“P”位）处与细菌核糖体的50S亚基成可逆性结合，阻断了转移核糖核酸（t-RNA）结合至“P”位上，同时也阻断了多肽链自受位（“A” 位）至“P”位的位移，因而细菌蛋白质合成受抑制。\n"
        "5.红霉素仅对分裂活跃的细菌有效。\n"
        "药代动力学: 1.静脉滴注后立即达血药浓度峰值，24小时内静滴2g，平均血药浓度为2.3～6.8mg/L，但个体差异较大。\n"
        "2.每12小时连续静脉滴注本品1g，则8小时后的血药浓度可维持于4～6mg/L。\n"
        "3.乳糖酸红霉素除脑脊液和脑组织外，广泛分布于各组织和体液中，尤以肝、胆汁和脾中的浓度为最高，在肾、肺等组织中的浓度可高出血药浓度数倍，在胆汁中的浓度可达血药浓度的10～40倍以上，在皮下组织、痰及支气管分泌物中的浓度也较高，痰中浓度与血药浓度相仿；在胸、腹水、脓液等中的浓度可达有效水平。\n"
        "4.本品有一定量（约为血药浓度的33%）进入前列腺及精囊中，但不易透过血脑屏障，脑膜有炎症时脑脊液中浓度仅为血药浓度的10%左右。\n"
        "5.可进入胎血和排入母乳中，胎儿血药浓度为母体血药浓度的5%～20%，母乳中药物浓度可达血药浓度的50%以上。\n"
        "6.表观分布容积为0.9L/kg，蛋白结合率为70%～90%。\n"
        "7.游离红霉素在肝内代谢，血半衰期为1.4～2小时，无尿患者的血半衰期可延长至4.8～6小时。\n"
        "8.红霉素主要在肝中浓缩和从胆汁排出，并进行肠肝循环，约2%～5%的口服量和10%～15%的注入量自肾小球滤过排除，尿中浓度可达10～100mg/L，粪便中也含有一定量。\n"
        "9.血或腹膜透析后极少被清除，故透析后无需加用。"
    )

    doc = Document(content=raw_text, meta={"title": "注射用乳糖酸红霉素"})
    
    print(f"Original Document Length: {len(raw_text)} chars")
    print(f"Original Document Word Count (approx space-split): {len(raw_text.split())}")
    print("-" * 50)

    # 2. Setup the Pipeline Components (Exact copy of your config)
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False
    )
    
    # CASE 1: Your current configuration
    print(">>> Testing CASE 1: split_by='passage' (Current Config)")
    splitter_passage = DocumentSplitter(
        split_by="passage", 
        split_length=100, 
        split_overlap=20
    )
    
    # Run Cleaner
    cleaned_docs = cleaner.run(documents=[doc])["documents"]
    print(f"After Cleaning: {len(cleaned_docs[0].content)} chars")
    
    # Run Splitter
    split_docs_passage = splitter_passage.run(documents=cleaned_docs)["documents"]
    print(f"Resulting chunks: {len(split_docs_passage)}")
    for i, d in enumerate(split_docs_passage):
        print(f"  Chunk {i+1} Length: {len(d.content)} chars | Content Preview: {d.content[:50]}...")
    
    print("-" * 50)

    # CASE 2: Testing split_by='word'
    print(">>> Testing CASE 2: split_by='word'")
    splitter_word = DocumentSplitter(
        split_by="word", 
        split_length=100, 
        split_overlap=20
    )
    
    split_docs_word = splitter_word.run(documents=cleaned_docs)["documents"]
    print(f"Resulting chunks: {len(split_docs_word)}")
    for i, d in enumerate(split_docs_word):
        print(f"  Chunk {i+1} Length: {len(d.content)} chars | Content Preview: {d.content[:50]}...")

    print("-" * 50)

    # CASE 3: Testing split_by='sentence'
    print(">>> Testing CASE 3: split_by='sentence'")
    splitter_sentence = DocumentSplitter(
        split_by="sentence", 
        split_length=6, # 5 sentences per chunk
        split_overlap=2
    )
    
    split_docs_sentence = splitter_sentence.run(documents=cleaned_docs)["documents"]
    print(f"Resulting chunks: {len(split_docs_sentence)}")
    for i, d in enumerate(split_docs_sentence):
        print(f"  Chunk {i+1} Length: {len(d.content)} chars | Content Preview: {d.content[:-1]}...")

if __name__ == "__main__":
    test_splitter_behavior()
