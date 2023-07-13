import hydra
import swifter
import json
from tqdm import tqdm
from pshmodule.utils import filemanager as fm
import mbti_prompt as mp
import pandas as pd
import openai


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    try:
        # Data Load
        df = fm.load(cfg.PATH.for_gpt)
        df_ref = fm.load(cfg.PATH.EXT_GPT_MARKET)

        # OpenAI Api Key
        openai.api_key = cfg.OPENAI.OPENAI_API_KEY
        model = "gpt-3.5-turbo"

        def generate_response(messages: list) -> str:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]

        # GPT Extraction
        predict = []
        start = df_ref.shape[0]
        print(f"start: {start}")
        df_temp = df.iloc[start:]
        print(df_temp.head())
        print(f"{start}번째 행부터 시작!")
        data = [[i[1]["type"], i[1]["input"]] for i in df_temp.iterrows()]
        for k, i in tqdm(enumerate(data)):
            messages = [
                {
                    "role": "system",
                    "content": mp.mbti["marketing"],
                },
                {
                    "role": "user",
                    "content": f"""{i[1]}\n
                                    위 문장에서 아래 정보를 ',\n'를 기준으로 뽑아줘.
                                    마케팅 주체 : ,\n
                                    타겟 : ,\n
                                    혜택 지급 조건 : ,\n
                                    혜택 : ,\n
                                    할인 수치 : ,\n
                                    프로모션 품목 : ,\n
                                    프로모션 장소 : ,\n
                                    이벤트 기간 : ,\n
                                    요일 정보 : ,\n
                                    시즌 정보 : 
                                """,
                },
            ]
            result = generate_response(messages)
            print(f"{start + k}번째 행")
            print(f"원문 : {i[1]}")
            print(f"답변 : {result}")
            print("-" * 100)
            predict.append(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{start + k}번째 행에서 오류!")

    finally:
        # Processing
        df_temp = df.iloc[start : start + len(predict)].copy()
        df_temp["marketing"] = predict
        df_temp["marketing_temp"] = df_temp.marketing.apply(
            lambda x: x.replace("마케팅 주체 : ", "")
            .replace("타겟 : ", "")
            .replace("혜택 지급 조건 : ", "")
            .replace("혜택 : ", "")
            .replace("할인 수치 : ", "")
            .replace("프로모션 품목 : ", "")
            .replace("프로모션 장소 : ", "")
            .replace("이벤트 기간 : ", "")
            .replace("요일 정보 : ", "")
            .replace("시즌 정보 : ", "")
        )
        df_temp["marketing_temp"] = df_temp.marketing_temp.apply(
            lambda x: x.replace("마케팅 주체: ", "")
            .replace("타겟: ", "")
            .replace("혜택 지급 조건: ", "")
            .replace("혜택: ", "")
            .replace("할인 수치: ", "")
            .replace("프로모션 품목: ", "")
            .replace("프로모션 장소: ", "")
            .replace("이벤트 기간: ", "")
            .replace("요일 정보: ", "")
            .replace("시즌 정보: ", "")
        )
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace("\n\n", "\n")
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace("\n", ",\n")
        df_temp["marketing_temp"] = df_temp.marketing_temp.str.replace(",,", ",")
        df_split_marketing = df_temp.marketing_temp.str.split(",\n", expand=True)
        df_split_marketing_true = df_split_marketing[
            df_split_marketing[0].str.contains(",\n")
        ]
        df_split_marketing_true = df_split_marketing_true[0].str.split(
            ",\n", expand=True
        )
        df_split_marketing_false = df_split_marketing[
            ~df_split_marketing[0].str.contains(",\n")
        ]
        df_split_marketing_temp = pd.concat(
            [df_split_marketing_false, df_split_marketing_true]
        )
        # 마케팅 주체: marketing_entity, 타겟: marketing_target, 혜택 지급 조건: benefit_conditions, 혜택: benefits, 할인 수치: discount_figure, 프로모션 품목: promotional_items, 프로모션 장소: promotional_place, 이벤트 기간: event_period, 요일 정보: dow_information, 시즌 정보: season_information, 소구점: solicitation_point
        df_split_marketing_temp.rename(
            columns={
                0: "marketing_entity",
                1: "marketing_target",
                2: "benefit_conditions",
                3: "benefits",
                4: "discount_figure",
                5: "promotional_items",
                6: "promotional_place",
                7: "event_period",
                8: "dow_information",
                9: "season_information",
            },
            inplace=True,
        )
        df_temp.rename(columns={"input": "label"}, inplace=True)
        df_temp = df_temp[["type", "label"]]
        df_result = pd.concat([df_split_marketing_temp, df_temp], axis=1)
        df_result.sort_index(ascending=True, inplace=True)
        df_result.fillna("없음", inplace=True)

        # save
        temp_dict = [
            {
                "marketing_entity": row["marketing_entity"].strip(),
                "marketing_target": row["marketing_target"].strip(),
                "benefit_conditions": row["benefit_conditions"].strip(),
                "benefits": row["benefits"].strip(),
                "discount_figure": row["discount_figure"].strip(),
                "promotional_items": row["promotional_items"].strip(),
                "promotional_place": row["promotional_place"].strip(),
                "event_period": row["event_period"].strip(),
                "dow_information": row["dow_information"].strip(),
                "season_information": row["season_information"].strip(),
                "type": row["type"],
                "label": row["label"].strip(),
            }
            for _, row in df_result.iterrows()
        ]
        with open(eval(f"cfg.PATH.EXT_GPT_MARKET"), "a", encoding="utf-8") as f:
            for line in temp_dict:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")


if __name__ == "__main__":
    main()
