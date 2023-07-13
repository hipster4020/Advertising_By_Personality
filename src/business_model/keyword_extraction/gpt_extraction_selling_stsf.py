import hydra
import json
import time
import math
import mbti_prompt as mp
import pandas as pd
import openai
from pshmodule.utils import filemanager as fm


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    start = 0
    sell = []
    sell_sent = []

    try:
        # Data Load
        # data = []
        # with open(cfg.PATH.for_selling_stsf, "r", encoding="utf-8") as f:
        #     for line in f:
        #         data.append(json.loads(line.rstrip("\n|\r")))
        # df = pd.DataFrame(data)
        df = fm.load(cfg.PATH.st_300)
        df_ref = fm.load(cfg.PATH.EXT_GPT_SELL)
        print(df.head())
        print(df.shape)

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
        start = df_ref.shape[0]
        df_start = df.iloc[start:].copy()
        s_time = time.time()
        print(f"start: {start}")
        print(f"{start}번째 행부터 시작!")

        for k, i in enumerate(df_start.iterrows()):
            # selling_point
            title = i[1]["title"]
            content = i[1]["content"]
            messages = [
                {
                    "role": "system",
                    "content": "너는 광고문구에 적용할 만한 소구점을 찾아내는 카피라이터야.",
                },
                {
                    "role": "user",
                    "content": f"""{mp.mbti["SELLING_ST"]}\n<원문>\n{title} {content}""",
                },
            ]

            result1 = generate_response(messages, model1)
            print(f"{start + k}번째 행")
            print(f"제목 : {title}, 본문 : {content}")
            print(f"답변 : {result1}")
            print("-" * 100)
            sell.append(result1.replace("<소구점>\n", ""))

            # selling_sent
            messages = [
                {
                    "role": "system",
                    "content": "너는 고객의 관심을 유도하고 이목을 집중시킬 광고문구를 만드는 카피라이터야.",
                },
                {
                    "role": "user",
                    "content": f"""{mp.mbti["SELLING_SENT_ST"]}\n<원문>\n{title} {content}\n{result1}\n<조건에 맞는 광고문구>""",
                },
            ]

            result2 = generate_response(messages, model2)
            print(f"{start + k}번째 행")
            print(f"제목 : {title}, 본문 : {content}")
            print(f"답변 : {result2}")
            print("-" * 100)
            sell_sent.append(
                result2.replace("본문: ", "\\\\")
                .replace("제목: ", "")
                .replace("\n<본문>\n", "\\\\")
                .replace("<제목>\n", "")
                .replace("\n<본문> : ", "\\\\")
            )

            # 1 row save
            temp_dict = [
                {
                    "no": i[1]["no"],
                    "origin_no": i[1]["origin_no"],
                    "origin": i[1]["origin"],
                    "st_sent": i[1]["title"] + "\\\\" + i[1]["content"],
                    "sell_point": result1.replace("<소구점>\n", ""),
                    "sell_sent": result2.replace("\n본문: ", "\\\\")
                    .replace("제목: ", "")
                    .replace("\n<본문>\n", "\\\\")
                    .replace("<제목>\n", "")
                    .replace("\n<본문> : ", "\\\\")
                    .replace("\n<본문>: ", "\\\\")
                    .replace("\n본문:", "\\\\"),
                }
            ]
            with open(cfg.PATH.EXT_GPT_SELL, "a", encoding="utf-8") as f:
                for line in temp_dict:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(json_record + "\n")
            print(f"{start + k}. Save Done!")
            print("-" * 100)

        math.factorial(100000)
        e_time = time.time()
        print(f"{e_time - s_time:.5f} sec")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # temp save
        df_temp = df.iloc[start : start + len(sell_sent)].copy()
        df_temp["sell_point"] = sell
        df_temp["sell_sent"] = sell_sent

        # Excel Save
        df_temp = df_temp[
            ["no", "origin_no", "origin", "title", "content", "sell_point", "sell_sent"]
        ]
        df_temp.rename(
            columns={
                "no": "번호",
                "origin_no": "원문 관리번호",
                "origin": "원문",
                "title": "ST 제목",
                "content": "ST 본문",
                "sell_point": "소구점",
                "sell_sent": "소구점이 반영된 광고 문구",
            },
            inplace=True,
        )
        fm.save(cfg.PATH.temp_save, df_temp)
        print("Save Done!")


if __name__ == "__main__":
    main()
