import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris

'''
# interactive 하게 데이터를 보여주는 페이지
# 일종의 Dashboard
'''

iris_dataset = load_iris()
#print(iris_dataset)

# 데이터 프레임으로 변환
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
print(df.head())

# 품종 정보도 추가
df['species'] = iris_dataset.target
#print(df.head())

species_str = {0 : 'setosa', 1 : 'versicolor', 2 : 'virginica'}

def map_species(x):
    return species_str[x]

df['species'] = df['species'].apply(map_species)

#print(df.head())
print(df.tail())

'''
# streamlit에서 데이터 프레임을 표현하는 방식은 table, dataframe 두 가지 사용 가능
'''
# --------------------------------------------------
#st.table(df.head())
#st.dataframe(df.tail())
# --------------------------------------------------

# 1. Select Box
# sidebar에 select box를 두고 종을 선택하게 한 다음 그에 해당하는 행만 추출
#st.sidebar.title('Iris Species')

# 클라이언트가 선택한 값이 지정
#selectedSpecies = st.sidebar.selectbox(
#    '확인하고 싶은 품종을 선택하세요', ['setosa', 'versicolor', 'virginica']
#)

# 지정된 값을 기반으로 원본 프레임을 필터링해서 출력 데이터 프레임으로 보여주기
#resultDF = (df[df['species'] == selectedSpecies])

#st.table(resultDF.head())

# --------------------------------------------------
# [실습] 여러 값을 선택할 수 있는 selectbox => multiselect
# multiselect는 return을 list로 해줌
# => 선택한 여러 종을 화면에 출력
#st.sidebar.title('Iris Species')
#multi_selectedSpecies = st.sidebar.multiselect(
#    '확인하고 싶은 품종을 선택하세요(복수 선택 가능)', ['setosa', 'versicolor', 'virginica']
#)

#print(multi_selectedSpecies)

#resultDF = (df[df['species'].isin(multi_selectedSpecies)])
#st.dataframe(resultDF)

# --------------------------------------------------
# Radio / Slider
st.sidebar.title('Iris Species')

# 클라이언트가 선택한 품종을 지정
multi_selectedSpecies = st.sidebar.multiselect(
    '확인하고 싶은 품종을 선택하세요(복수 선택 가능)', ['setosa', 'versicolor', 'virginica']
)

# 선택된 품종에서 기준 컬럼을 선택
selectRadio = st.sidebar.radio(
    '기준 컬럼을 선택하세요',
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    horizontal = True
)

# 선택한 컬럼의 값의 범위를 지정 - slider
selectRange = st.sidebar.slider(
    '기준 컬럼으로 값의 범위를 지정하세요',
    0.0,            # 시작 값
    10.0,           # 끝 값
    #(2.5, 7.5)      # 선택 범위, 선택 범위를 하나만 지정 (value=2.5)
    value=2.5       # 선택 범위를 하나만 지정 (ex. value=2.5) 처음부터 선택값까지
)

# 선택이 모두 끝나면 실행 될 버튼
submitBtn = st.sidebar.button(
    '결과 확인'
)

# slider 선택 범위 : slider_range (리스트 형식으로 2개의 값[최소값, 최대값]이 저장)
# slider_range[0] : 최소값
# slider_range[1] : 최대값

# 버튼이 눌러지면 결과 확인
if submitBtn:
    # multi selectbox filtering
    resultDF = df[df['species'].isin(multi_selectedSpecies)]
    # radio / slider filtering
    resultDF = resultDF[(resultDF[selectRadio] >= selectRange[0]) & (resultDF[selectRadio] <= selectRange[1])]
    st.table(resultDF)
