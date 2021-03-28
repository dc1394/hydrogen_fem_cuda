#
# プログラム名
#
PROG = hydrogen_fem

#
# ソースコードが存在する相対パス
#
VPATH = src/hydrogen_fem

#
# コンパイル対象のソースファイル群
#
SRCS := hydrogen_fem_main.cpp hydrogen_fem.cpp sygvd_cuda.cu 

#
# ターゲットファイルを生成するために利用するオブジェクトファイル
#
OBJS := hydrogen_fem_main.o hydrogen_fem.o sygvd_cuda.o

#
# *.cppファイルの依存関係が書かれた*.dファイル
#
DEPS = $(OBJS:.o=.d)

#
# C++コンパイラの指定
#
CXX = g++

#
# C++コンパイラの指定
#
CU = nvcc

#
# C++コンパイラに与える、（最適化等の）オプション
#
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -mtune=native -march=native -fopenmp

#
# C++コンパイラに与える、（最適化等の）オプション
#
CUFLAGS = -I/usr/local/cuda/include -O3 -std=c++17

#
# リンク対象に含めるライブラリの指定
#
LDFLAGS = -L/usr/local/cuda/lib64 -lcusolver -lcudart -lm -ldl

#
# makeの動作
#
all: $(PROG) ; rm -f $(OBJS) $(DEPS)

#
# 依存関係を解決するためのinclude文
#
-include $(DEPS)

#
# プログラムのリンク
#
$(PROG): $(OBJS)
		$(CXX) $^ $(LDFLAGS) $(CXXFLAGS) -o $@

#
# プログラムのコンパイル
#
%.o: %.cpp
		$(CXX) $(CXXFLAGS) -c -MMD -MP $<

#
# プログラムのコンパイル
#
%.o: %.cu
		$(CU) $(CUFLAGS) -c -MMD -MP $<

#
# make cleanの動作
#
clean:
		rm -f $(PROG) $(OBJS) $(DEPS)