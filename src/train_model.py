from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import os
import torch
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

def train_model(dataset, model_name, output_dir="modelo_treinado", max_length: int = 384):
    os.makedirs(output_dir, exist_ok=True)

    print(f"DEBUG: dataset original tem {len(dataset)} samples")
    if len(dataset) > 0:
        print(f"DEBUG: primeiro sample: {dataset[0]}")
        print(f"DEBUG: colunas: {dataset.column_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: device = {device}")

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        # Tokenize question + context com offsets para alinhar respostas
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        starts = []
        ends = []
        not_found = 0

        # tokenized fields são listas de listas quando em batch
        for i in range(len(tokenized["input_ids"])):
            offsets = tokenized["offset_mapping"][i]
            token_type_ids = tokenized.get("token_type_ids", [None] * len(offsets))[i]
            ctx = examples["context"][i]
            ans = examples["answer"][i] if "answer" in examples else ""

            # encontra char span da resposta no contexto
            answer_start_char = ctx.find(ans) if ans else -1
            if answer_start_char == -1 or not ans:
                # fallback: marca como não encontrado
                starts.append(0)
                ends.append(0)
                not_found += 1
            else:
                answer_end_char = answer_start_char + len(ans)

                # encontra indices de token que cobrem a resposta
                token_start_index = None
                token_end_index = None
                
                for idx, (off, tt) in enumerate(zip(offsets, token_type_ids)):
                    if tt != 1:  # pula tokens de pergunta e especiais
                        continue
                    token_char_start, token_char_end = off
                    
                    if token_char_start <= answer_start_char < token_char_end:
                        token_start_index = idx
                    if token_char_start < answer_end_char <= token_char_end:
                        token_end_index = idx
                    
                    if token_start_index is None and token_char_start > answer_start_char:
                        token_start_index = idx
                
                # se end index não encontrado, acha último token
                if token_end_index is None:
                    for idx in range(len(offsets) - 1, -1, -1):
                        if token_type_ids[idx] == 1 and offsets[idx][1] <= answer_end_char:
                            token_end_index = idx
                            break

                if token_start_index is None or token_end_index is None:
                    starts.append(0)
                    ends.append(0)
                    not_found += 1
                else:
                    starts.append(token_start_index)
                    ends.append(token_end_index)

        # remove offset mappings (não necessário para treinamento)
        tokenized.pop("offset_mapping", None)

        tokenized["start_positions"] = starts
        tokenized["end_positions"] = ends

        if not_found:
            print(f"  ⚠️ preprocess: {not_found}/{len(starts)} exemplos sem resposta encontrada")

        return tokenized

    # map e remove colunas originais
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    print(f"DEBUG: dataset tokenizado tem {len(tokenized)} samples")
    if len(tokenized) > 0:
        print(f"DEBUG: primeiro sample tokenizado keys: {list(tokenized[0].keys())}")
    else:
        print("❌ ERRO: Dataset tokenizado está vazio!")
        return

    # safe defaults para CPU vs GPU
    per_device_batch = 8 if device == "cuda" else 2
    dataloader_pin_memory = True if device == "cuda" else False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=3e-5,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=dataloader_pin_memory,
        fp16=(device == "cuda"),
        save_strategy="no",  # Não salva checkpoints intermediários
        logging_steps=100,
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    try:
        print("🔹 Iniciando treinamento...")
        trainer.train()
        print("✅ Treinamento concluído com sucesso!")
    except KeyboardInterrupt:
        print("⚠️ Treinamento interrompido pelo usuário. Salvando modelo...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        # Tenta salvar mesmo assim
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("✓ Modelo salvo apesar do erro")
        except Exception as save_err:
            print(f"❌ Falha ao salvar modelo: {save_err}")
        raise

    # Salva modelo final
    print("💾 Salvando modelo final...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modelo salvo em: {output_dir}")