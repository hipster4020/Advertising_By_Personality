import hydra
import time
import math
import mbti_prompt as mp
import openai
from pshmodule.utils import filemanager as fm


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    start = 0
    sell = []
    sell_sent = []

    try:
        # Data Load
        df = fm.load(cfg.PATH.test)
        # df_ref = fm.load(cfg.PATH.EXT_GPT_SELL)

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model1 = "gpt-3.5-turbo"
        model2 = "gpt-4"

        def generate_response(messages: list, model: str) -> str:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                # temperature=temperature,
            )
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        # start = df_ref.shape[0]
        # start = 0
        # df_start = df.iloc[start:].copy()
        s_time = time.time()

        # print(f"start: {start}")
        # print(df_start.head())
        # print(f"{start}번째 행부터 시작!")

        for k, i in enumerate(df.iterrows()):
            title = i[1]["title"]
            content = i[1]["content"]

            # selling_point
            messages = [
                {
                    "role": "system",
                    "content": "너는 광고문구에 적용할 만한 소구점을 찾아내는 카피라이터야.",
                },
                {
                    "role": "user",
                    "content": f"""{mp.mbti["SELLING_NTNF"]}\n<원문>\n{title} {content}""",
                },
            ]

            result1 = generate_response(messages, model1)
            print(f"{start + k}번째 행")
            print(f"제목 : {i[1]['title']}, 본문 : {i[1]['content']}")
            print(f"답변 : {result1}")
            print("\n")
            sell.append(result1.replace("<소구점>\n", ""))

            # temp = result1.split("<소구점>")
            # sell.append(temp[1])
            # sell_sent.append(temp[0])
            # result_json1 = json.loads(result1)

            # temp = result1.split("<소구점>")
            # sell.append(temp[1])
            # sell_sent.append(temp[0].replace("<결과>", ""))

            # selling_sent
            messages = [
                {
                    "role": "system",
                    "content": "너는 고객의 관심을 유도하고 이목을 집중시킬 광고문구를 만드는 카피라이터야.",
                },
                {
                    "role": "user",
                    "content": f"""{mp.mbti["SELLING_SENT_NTNF"]}<원문>\n{title} {content}\n<조건에 맞는 광고문구>
                                   {result1}
                                    """,
                },
            ]
            result2 = generate_response(messages, model2)
            print(f"{start + k}번째 행")
            print(f"제목 : {title}, 본문 : {content}")
            print(f"답변 : {result2}")

            temp2 = (
                result2.replace("본문: ", "\\\\")
                .replace("제목: ", "")
                .replace("<본문>: ", "\\\\")
                .replace("<제목>: ", "")
                .replace("<조건에 맞는 광고문구>\n", "")
                .replace("<광고문구>\n", "")
                .replace("<변환된 광고문구>\n", "")
                .replace("<변환 후 광고문구>\n", "")
                .replace("<재생성된 광고문구>\n", "")
            )
            sell_sent.append(temp2)

            # # 1 row save
            # temp_dict = [
            #     {
            #         "origin": i[1]["title"] + "\\\\" + i[1]["content"],
            #         "sell_point": result1.replace("<소구점>\n", ""),
            #         "sell_sent": temp2,
            #     }
            # ]
            # with open(cfg.PATH.EXT_GPT_SELL, "a", encoding="utf-8") as f:
            #     for line in temp_dict:
            #         json_record = json.dumps(line, ensure_ascii=False)
            #         f.write(json_record + "\n")
            # print(f"{start + k}. Save Done!")
            # print("-" * 100)

        math.factorial(100000)
        e_time = time.time()
        print(f"{e_time - s_time:.5f} sec")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류")

    finally:
        # temp save
        df_temp = df.iloc[start : start + len(sell)].copy()
        print(df_temp.head())

        df_temp["origin"] = df_temp["title"] + "\\\\" + df_temp["content"]
        df_temp["sell_point"] = sell
        df_temp["sell_sent"] = sell_sent

        # Excel Save
        df_temp = df_temp[["origin", "sell_point", "sell_sent"]]
        # df_temp = df_temp[["label", "use_market", "sell_point", "sell_sent"]]
        df_temp.rename(
            columns={
                "origin": "원문",
                "sell_point": "소구점",
                "sell_sent": "광고 문구",
            },
            inplace=True,
        )
        fm.save(cfg.PATH.temp_save, df_temp)


if __name__ == "__main__":
    main()
