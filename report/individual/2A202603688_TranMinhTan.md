# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác

SEED: **1281**. Thử thách #4: kNN với k=1,5,25.

## 1. Bằng chứng và phạm vi đánh giá

Bài nộp gồm bốn tệp:

- [Notebook đã chạy, có output](day01_lab.ipynb).
- [Kết quả kiểm tra và số liệu JSON](lab1_report.json).
- [Ảnh tính tay có các bước](hand_calc.jpg).
- Báo cáo Markdown này.


Kiểm tra đã ghi: 15/15 đạt. Kiểm tra chưa đạt: không. Dự đoán thiếu: không. Nhóm model chưa có kết quả: không.

Bài có AI hỗ trợ code, dự đoán tham khảo, giải thích và tạo báo cáo; chi tiết ở AI log. So sánh LLM là Lite 3.5 với Lite 3.1, không phải hai model lớn/nhỏ như cấu hình ban đầu của đề.

Dữ liệu sau bỏ trùng: 5169 tin, gồm 4516 ham và 653 spam. Train=3619, validation=775, test=775, SUB=40. SUB gồm 20 spam và 20 ham lấy từ test.

## 2. Kết quả chính

| Model | Precision | Recall | F1 test (*) | F1 SUB | Train s | Tin/s | p95 ms | USD/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.9880 | 0.7961 | 0.8817 | 0.8235 | 3.7265 | 51220.6475 | 0.0751 | 0.0000 |
| 2. BoW + MLP (deep learning) | 0.9688 | 0.9029 | 0.9347 | 0.8889 | 9.7224 | 22408.1951 | 0.2956 | 0.0000 |
| 3a. Encoder (MiniLM) + kNN | 0.9151 | 0.9417 | 0.9282 | 0.9189 | 83.3528 | 40.2807 | 58.6031 | 0.0000 |
| 3b. Encoder (MiniLM) + LogReg | 0.9515 | 0.9515 | 0.9515 | 0.9189 | 83.8003 | 53.7065 | 39.0323 | 0.0000 |
| 3c. Encoder (MiniLM) + MLP head | 0.9898 | 0.9417 | 0.9652 | 0.9189 | 84.6203 | 51.4463 | 53.7260 | 0.0000 |
| 4a. Lite 3.1 zero-shot (đối chiếu) | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.0000 | 0.1940 | 7074.7536 | 0.0172 |
| 4b. Lite 3.5 zero-shot (chính) | 0.8261 | 0.9500 | 0.8837 | 0.8837 | 0.0000 | 0.1858 | 1704.7588 | 0.0216 |
| 4c. Lite 3.5 few-shot | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.0000 | 0.2382 | 1247.1321 | 0.0560 |

(*) LLM chỉ đánh giá trên SUB; F1 ở hàng LLM không phải F1 toàn bộ test. So sánh công bằng giữa các phương pháp bằng F1 SUB. Độ trễ đo từ văn bản thô đến nhãn; LLM p50/p95 không gồm thời gian chờ quota chủ động, throughput cả lô có gồm chờ. USD là ước tính theo số từ và giá tham khảo, chưa gồm đầy đủ token suy luận và không phải hóa đơn free tier.

## 3. Dự đoán, thực tế và giải thích Phần 0–4

### Câu 0.1 — Data leakage

SMS trùng ở cả train/test làm test chứa nội dung đã gặp khi học, có thể làm điểm lạc quan. Bỏ trùng trước khi chia giúp hạn chế dạng rò rỉ đó; các mẫu chỉ khác số điện thoại vẫn có thể còn tương tự nhau.

### ML và learning rate

Dự đoán ML là F1 khoảng 0.80; F1 test thực tế=0.8817, lệch +8.17 điểm phần trăm. Precision=0.9880, recall=0.7961; tỷ lệ spam bị bỏ sót xấp xỉ 20.39%.

Khi tin spam chứa free bị đoán thiếu, p-y âm; đóng góp gradient của tin này làm trọng số free có xu hướng tăng sau bước trừ gradient. Cập nhật thực tế là trung bình của tất cả mẫu train.

Trong lượt Colab đã cung cấp, loss cuối ở lr=0.001/0.5/50 lần lượt khoảng 0.627/0.084/0.006. lr=50 tăng vọt ban đầu rồi giảm, nên không phân kỳ kéo dài. Đây là loss train, chưa đủ chọn model tốt nhất trên dữ liệu mới.

**Bắt lỗi 1:** train và đánh giá trên cùng dữ liệu, không có test độc lập; dữ liệu chưa bỏ trùng và vocab xây từ toàn bộ raw làm phương pháp đánh giá thiếu tin cậy. Accuracy cao không cho biết bắt được bao nhiêu spam.

Một lỗi ML thực tế: `thesmszone.com lets you send free anonymous and masked messages..im sending this message from there..do you see the potential for abuse???`; nhãn thật=spam, dự đoán=ham. BoW cộng các trọng số token, bỏ thứ tự và ngữ cảnh; các tín hiệu học được ở tin này không đủ để ra nhãn đúng. Không thể xác nhận nguyên nhân riêng từng token nếu chưa phân rã điểm số.

### Deep Learning

MLP F1 test=0.9347, LogReg=0.8817, chênh +5.30 điểm phần trăm. Validation loss nhỏ nhất ở epoch 6 (đánh số từ 1); đây là mốc chọn checkpoint theo validation. Dấu hiệu tăng validation loss trong lúc train loss giảm xuất hiện sớm nhất ở epoch 7; một nhịp tăng riêng lẻ chưa chứng minh overfit kéo dài. Dữ liệu nhỏ và đầu vào BoW giới hạn thông tin; thêm tham số không bảo đảm thắng rõ.

Bỏ ReLU: F1 test=0.9239, so với MLP có ReLU=0.9347. Hai lớp affine liên tiếp gộp thành một lớp affine, nên ranh giới phân loại vẫn tuyến tính; quá trình tối ưu có thể khác LogReg trực tiếp.

MLP 2000→64→1 có 2000×64+64+64×1+1=128129 tham số; LogReg có 2001. zero_grad xóa gradient cũ; forward tính logits; loss so với nhãn; backward tính gradient; step cập nhật tham số. Quên zero_grad làm gradient cộng dồn qua minibatch, khác thuật toán dự định.

### Embedding

Cùng encoder, LogReg F1=0.9515, MLP head F1=0.9652; thêm hidden layer thay đổi +1.37 điểm phần trăm. Encoder đóng băng đã cung cấp đặc trưng ngữ nghĩa, nên lợi ích thêm phi tuyến phải đo chứ không mặc định lớn hơn.

Sáu tin ngụy trang: BoW đúng 4/6, Encoder+LogReg đúng 5/6.

Ví dụ hai cách cho kết quả khác nhau: `Exclusive: get a brand new phone at no cost, just reply with your address`; nhãn thật=1, BoW=0, embedding=1. BoW dựa vào token riêng lẻ; embedding dùng biểu diễn ngữ nghĩa đã học từ trước.

Encoder không được train lại để lọc spam; biểu diễn ngôn ngữ có thể giúp gom mẫu tương đồng, nhưng không bảo đảm hai lớp tách tuyệt đối. Encoder giúp giảm lượng nhãn riêng cho tác vụ so với học biểu diễn từ đầu. kNN lưu vector train rồi bỏ phiếu từ hàng xóm; tìm vét cạn sẽ tốn tài nguyên khi dữ liệu lớn.

**Bắt lỗi 2:** M@q chưa phải cosine nếu chưa chuẩn hóa; argsort tăng dần rồi lấy đầu chọn điểm thấp nhất thay vì cao nhất.

### LLM

Lite 3.1 zero-shot F1 SUB=0.9000, Lite 3.5=0.8837; Lite 3.5 few-shot=0.9000, đổi +1.63 điểm phần trăm so với zero-shot cùng model. Chi phí ước tính zero/few-shot lần lượt $0.02160/$0.05600 mỗi 1000 tin.

Tỷ lệ chi phí few/zero-shot=2.59 lần. Thêm ví dụ làm prompt dài hơn, nhưng chất lượng vẫn phụ thuộc ví dụ và dữ liệu.

Embedding+LogReg F1 SUB=0.9189; chênh Lite 3.5 zero-shot với embedding=-3.52 điểm phần trăm. Tập 40 tin nhỏ nên không xếp hạng chắc chắn từ chênh lệch nhỏ.

Temperature=0 giảm biến thiên lấy mẫu, không bảo đảm tái lập tuyệt đối. max_tokens=1 có thể cắt nhãn nhiều token; thử kết nối đã ghi spam dùng 2 token. Model suy luận cần thêm ngân sách, nên Gemini dùng 512 token thay vì 3. Trong cấu hình OpenAI của đề, không tắt suy luận có thể dùng hết ngân sách trước khi sinh nhãn. Zero-shot hữu ích khi tác vụ mới chưa có dữ liệu có nhãn để train.

**Bắt lỗi 3:** chuỗi not spam vẫn chứa spam; kiểm tra substring sẽ đảo nhãn. Đầu ra không rõ cần ghi nhận invalid thay vì coi là nhãn hợp lệ.

### Dự đoán đã ghi trước thí nghiệm

| Key | Thời gian runtime | Dự đoán |
| --- | --- | --- |
| ml_f1 | 08:03:41 | Dự đoán F1 khoảng 0.80, precision cao hơn recall vì từ khóa giúp nhận diện spam rõ ràng nhưng có thể bỏ sót spam né từ khóa. |
| ml_lr | 08:03:45 | Với lr=0.001, tôi dự đoán loss giảm chậm vì mỗi bước cập nhật trọng số rất nhỏ. Với lr=50, loss có thể dao động hoặc tăng vì bước cập nhật quá lớn, nhưng cũng có thể giảm nhanh lúc đầu. |
| dl_curve | 08:03:58 | Validation loss có thể giảm rồi tăng do overfitting; tôi đoán MLP có F1 gần hoặc nhỉnh hơn Logistic Regression, không chắc thắng rõ trên BoW. |
| dl_relu | 08:04:08 | Bỏ ReLU làm hai lớp affine gộp thành một lớp affine; tôi đoán F1 gần Logistic Regression và có thể giảm so với MLP có ReLU. |
| emb_search | 08:06:26 | Tôi đoán các tin gần truy vấn về nghĩa sẽ thiên về spam dù không trùng các từ free, win hoặc prize. |
| emb_tricky | 08:07:21 | Tôi đoán Encoder+LogReg đúng nhiều hơn BoW trên sáu tin ngụy trang, vì có thể nhận ra ngữ nghĩa tương tự khi thay cách diễn đạt. |
| llm_models | 08:07:23 | Tôi đoán Lite 3.5 và Lite 3.1 có F1 SUB gần nhau, có thể 3.5 nhỉnh hơn nhưng không chắc; giá tham khảo 3.5 cao hơn. LLM có thể gần hoặc hơn embedding, cần đo trên cùng SUB. |
| llm_fewshot | 08:14:25 | Tôi đoán thêm bốn ví dụ giúp F1 giữ nguyên hoặc tăng nhẹ, nhưng cũng có thể không cải thiện trên 40 tin. Chi phí input tăng do prompt dài hơn, có thể vài lần. |
| tradeoff | 08:17:13 | Tôi đoán BoW+LogReg có thể nằm trên Pareto nhờ rất nhanh; embedding có thể thêm lựa chọn chất lượng cao hơn. LLM nhiều khả năng vượt SLA p95 50ms. |
| hard_eval | 08:17:13 | Tôi đoán accuracy BoW khoảng 50–70%, embedding 60–80%, LLM 70–90%; đây là giả thuyết trước chạy. BoW có thể khó với né từ khóa, LLM có thể khó với prompt injection và tin mơ hồ. |

Dự đoán có AI hỗ trợ và được ghi bằng predict() trước khi chạy các thí nghiệm; thời gian trong bảng là thời gian runtime thực tế.

## 4. Pareto và 50 case khó

Các phương án không bị lấn át theo F1 SUB và p95: 1. BoW + LogReg (tự code), 2. BoW + MLP (deep learning), 3b. Encoder (MiniLM) + LogReg.

Ví dụ: 3a. Encoder (MiniLM) + kNN bị 3b. Encoder (MiniLM) + LogReg lấn át theo hai số đo này.

Trong các phương án đạt p95<50ms, 3b. Encoder (MiniLM) + LogReg có F1 SUB cao nhất (0.9189), p95=39.032ms, throughput=53.71 tin/s. Chọn đây làm ứng viên ban đầu; cần load test và đánh giá trên phân phối thực tế trước triển khai.

Một lỗi trên 40 tin đổi accuracy 2.5 điểm phần trăm, nhưng tác động F1 phụ thuộc TP/FP/FN. Từ dự đoán hoàn hảo trên 20 spam/20 ham, một FN cho F1=38/39≈0.9744, một FP cho F1=40/41≈0.9756. Chênh nhỏ giữa các chấm chưa đủ bằng chứng về thứ hạng ổn định.

| Model | F1 test thường (SUB) | F1 50 case khó | acc 50 case khó |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.8235 | 0.3333 | 0.5200 |
| 2. BoW + MLP | 0.8889 | 0.4878 | 0.5800 |
| 3a. Emb + kNN | 0.9189 | 0.7368 | 0.7000 |
| 3b. Emb + LogReg | 0.9189 | 0.6923 | 0.6800 |
| 3c. Emb + MLP head | 0.9189 | 0.6800 | 0.6800 |
| 4b. Lite 3.5 zero-shot (chính) | 0.8837 | 1.0000 | 1.0000 |
| 4d. Lite 3.5 + prompt phòng thủ | 0.8837 | 1.0000 | 1.0000 |

Chênh giảm lớn nhất: 1. BoW + LogReg, +49.02 điểm phần trăm F1 (SUB trừ tập khó). Test ngẫu nhiên cùng nguồn dễ hơn tập cố tình né từ khóa, giả danh và gây nhiễu chỉ dẫn.

1. BoW + LogReg: nhóm có accuracy thấp nhất là Spam giọng tự nhiên, 10.0%.

2. BoW + MLP: nhóm có accuracy thấp nhất là Spam giọng tự nhiên, 10.0%.

3a. Emb + kNN: nhóm có accuracy thấp nhất là Nói về spam, 25.0%.

3b. Emb + LogReg: nhóm có accuracy thấp nhất là Spam giọng tự nhiên, 30.0%.

3c. Emb + MLP head: nhóm có accuracy thấp nhất là Thao túng / lừa đảo, 25.0%.

4b. Lite 3.5 zero-shot (chính): đúng 100% ở tất cả các nhóm của bộ test này; chưa xác định được nhóm khó nhất từ kết quả này.

4d. Lite 3.5 + prompt phòng thủ: đúng 100% ở tất cả các nhóm của bộ test này; chưa xác định được nhóm khó nhất từ kết quả này.

Case sai: #11, 1. BoW + LogReg, thật=spam; `Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it`. Bẫy: Giọng thân mật, giả làm người quen, thực chất là khuyến mãi.

Case sai: #8, 2. BoW + MLP, thật=ham; `Cash prize for the quiz night goes to the winning team, sign up with Sarah by Friday`. Bẫy: 'Cash prize' nhưng là sự kiện nội bộ.

Prompt injection: bình thường=100.0% đúng, phòng thủ=100.0% đúng. Các nhóm còn lại: bình thường=100.0%, phòng thủ=100.0%.

BoW chỉ thấy token nên nhạy với né từ khóa; embedding có thể nhầm ngữ nghĩa với ý định; LLM có thể bị chỉ dẫn nằm trong SMS đánh lừa. Số liệu nhóm trên là kết quả thực tế để đối chiếu các cơ chế này.

Phòng thủ ngoài prompt: kiểm tra domain/URL và luật rủi ro độc lập, giới hạn quyền của LLM và chuyển trường hợp nguy hiểm sang lớp kiểm tra khác. Chuẩn hóa leetspeak, Unicode và ký tự chèn có thể hỗ trợ BoW nhưng có thể làm hỏng mã OTP, tên riêng hoặc văn bản hợp lệ. 50 case viết tay dùng để tìm điểm yếu, không đại diện tần suất spam ngoài đời và không dùng thay tập đánh giá thực tế.

## 5. Tính tay và thử thách cá nhân

Bài tính: accuracy=4516/5169≈0.87; sigmoid(0)=0.50; F1=2×80/(2×80+20+40)≈0.73; cosine=4/(sqrt(5)×3)≈0.60; chi phí giả thiết của đề=100 triệu token input×0.10/triệu + 1 triệu output×0.50/triệu=$10.50/ngày. Các bước viết tay được lưu trong [hand_calc.jpg](hand_calc.jpg).

| k | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| 1 | 0.9794 | 0.9065 | 0.9417 | 0.9238 |
| 5 | 0.9806 | 0.9151 | 0.9417 | 0.9282 |
| 25 | 0.9716 | 0.8785 | 0.9126 | 0.8952 |

Trong k=1,5,25 đã chạy, F1 cao nhất thuộc k=5 (0.9282). k nhỏ nhạy với hàng xóm nhiễu; k lớn có thể trộn vùng khác lớp. Kết quả này là thí nghiệm trên test; khi triển khai nên chọn k trên validation và đánh giá cuối trên test độc lập. Code thử thách có AI hỗ trợ.

## 6. Phương án cho sản phẩm thật

| Model | USD/tháng 300 triệu SMS |
| --- | --- |
| 4a. Lite 3.1 zero-shot (đối chiếu) | 5167.5000 |
| 4b. Lite 3.5 zero-shot (chính) | 6481.0000 |
| 4c. Lite 3.5 few-shot | 16801.0000 |

10 triệu SMS/ngày tương đương khoảng 115.74 tin/s trung bình; còn cần đánh giá tải đỉnh và dự phòng. Router dùng model nhanh cho trường hợp chắc chắn, chuyển trường hợp lưỡng lự sang LLM; tỷ lệ chuyển tiếp giảm chi phí nhưng nhánh API có thể vi phạm SLA đồng bộ 50ms. Cần đo cả hệ thống; không suy từ p95 của riêng đầu phân loại.

Điểm rút ra từ thí nghiệm learning rate: lr lớn gây tăng loss ban đầu nhưng không nhất thiết phân kỳ. Chọn model dựa vào chất lượng, lỗi cụ thể, latency và chi phí; không mặc định model mới hơn hay nhiều tham số hơn sẽ tốt hơn.

Về năng lực xử lý, Encoder+LogReg đạt p95 39.032ms nhưng throughput đo được chỉ 53.71 tin/s, thấp hơn khoảng 115.74 tin/s trung bình cần cho 10 triệu SMS/ngày. Một cấu hình như trong thí nghiệm chưa đủ tải này; cần tăng năng lực xử lý, phân phối tải và load test. Đạt p95 từng request không đồng nghĩa đã đạt throughput toàn hệ thống.

## 7. AI log

| Phần | Đã hỏi AI | AI hỗ trợ | Kiểm chứng |
| --- | --- | --- | --- |
| Toàn bài | Đọc repo, hoàn thiện code và hướng dẫn nộp | 14 hàm TODO, code thử thách kNN, giải thích và bộ xuất báo cáo | Chạy các ô kiểm tra notebook và thực nghiệm trên Colab; đối chiếu output |
| Dự đoán/câu hỏi | Gợi ý nội dung predict, data leakage, gradient và learning rate | Gợi ý dự đoán trước thí nghiệm và diễn giải theo số đo thực tế | Đối chiếu thời gian predict với thí nghiệm; không coi gợi ý AI là suy nghĩ độc lập của học viên |
| Gemini | Chọn model theo quota, cài Colab Secret và chạy thử | Lite 3.5/3.1, giãn cách request và notebook thử 4 lần gọi | Đã tự thêm Secret; hai model trả nhãn hợp lệ cho hai SMS mẫu/model |

Báo cáo này được sinh từ biến runtime bằng code có AI hỗ trợ. Không khai là bài tự code hoặc thử thách thực hiện độc lập không dùng AI. Phản ánh được diễn giải từ các thí nghiệm đã chạy.
