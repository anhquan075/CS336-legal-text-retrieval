from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "hogger32/xlmRoberta-for-VietnameseQA"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Cá nhân cần có những điều kiện nào thì sẽ được tự đứng ra nhận khoán đất rừng?',
    'context': '''1. Việc triển khai các dự án bảo vệ và phát triển rừng, trồng rừng, khoanh nuôi tái sinh có trồng bổ sung cần được khoán ổn định cho người dân từ khâu trồng, chăm sóc, bảo vệ theo chu kỳ lâm sinh và hưởng lợi theo quy định của Quyết định số 38/2016/QĐ-TTg. 2. Chủ đầu tư ký hợp đồng trồng rừng, bảo vệ và phát triển rừng với hộ dân, nhóm hộ tham gia trồng rừng bao gồm hướng dẫn thiết kế kỹ thuật, dự toán công trình lâm sinh được duyệt. 3. Người dân khi được giao khoán đất trồng rừng, phát triển rừng có thể tự trồng rừng trước và được hỗ trợ sau theo định mức quy định. 4. Đối với đất lâm nghiệp đã giao cho người dân ổn định lâu dài, người dân được tự trồng rừng trước. Diện tích trồng thành rừng sẽ được ưu tiên đưa vào dự án trồng rừng, bảo vệ và phát triển rừng của tỉnh hỗ trợ sau đầu tư và hỗ trợ vốn theo quy định. 5. Danh sách các hộ tham gia trồng rừng, phát triển rừng được niêm yết công khai tại Ủy ban nhân dân xã, nhà văn hóa thôn, bản về diện tích và số tiền được nhận. 6. Đối với đất trồng rừng của các công ty nông, lâm nghiệp đã được cổ phần hóa theo quy định tại Nghị định số 118/2014/NĐ-CP ngày 17 tháng 12 năm 2014 của Chính phủ mà Nhà nước sở hữu trên 50% vốn điều lệ khi nhận hỗ trợ để thực hiện dự án trồng rừng thì khoán ổn định lâu dài cho hộ gia đình.'''
}
res = nlp(QA_input)
print(res)

