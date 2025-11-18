import os
import xml.etree.ElementTree as ET

def load_xml_data(xml_folder):
    """
    Carrega dados de XMLs no formato QAPairs.
    Estrutura esperada:
    <QAPairs>
        <QAPair pid="...">
            <Question qid="..." qtype="...">Pergunta?</Question>
            <Answer>Resposta...</Answer>
        </QAPair>
    </QAPairs>
    
    Retorna lista de dicts: {"question", "context", "answer"}
    """
    data = []
    
    if not os.path.exists(xml_folder):
        print(f"❌ ERRO: Diretório {xml_folder} não encontrado")
        return data
    
    xml_files = [f for f in os.listdir(xml_folder) if f.lower().endswith(".xml")]
    print(f"DEBUG: encontrados {len(xml_files)} arquivos XML em {xml_folder}")
    
    for filename in xml_files:
        path = os.path.join(xml_folder, filename)
        print(f"DEBUG: processando {filename}")
        
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Procura por QAPair (estrutura fornecida)
            qapairs = root.findall(".//QAPair")
            print(f"  └─ encontrados {len(qapairs)} QAPair")
            
            for qapair in qapairs:
                question_elem = qapair.find("Question")
                answer_elem = qapair.find("Answer")
                
                if question_elem is not None and answer_elem is not None:
                    question_text = question_elem.text
                    answer_text = answer_elem.text
                    
                    if question_text and answer_text:
                        # Limpa whitespace excessivo
                        question_text = " ".join(question_text.split())
                        answer_text = " ".join(answer_text.split())
                        
                        # Extrai primeira frase ou trecho da resposta como answer curto
                        answer_short = answer_text[:200]  # primeiros 200 chars como label
                        
                        data.append({
                            "question": question_text,
                            "context": answer_text,  # context é a resposta completa
                            "answer": answer_short   # answer é trecho da resposta (para QA)
                        })
                        print(f"    ✓ adicionado: {question_text[:60]}...")
        
        except ET.ParseError as e:
            print(f"  ❌ ERRO ao parsear XML {filename}: {e}")
        except Exception as e:
            print(f"  ❌ ERRO ao processar {filename}: {e}")
    
    print(f"\n✅ RESULTADO: {len(data)} samples extraídos\n")
    return data