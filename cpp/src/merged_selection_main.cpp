#include "solidattention/selection.hpp"
#include "solidattention/uring_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstdlib>

template<class T> std::vector<T> read_all(const std::filesystem::path& path,std::size_t count){std::vector<T> value(count);std::ifstream in(path,std::ios::binary);if(!in||!in.read(reinterpret_cast<char*>(value.data()),count*sizeof(T)))throw std::runtime_error("short input: "+path.string());return value;}
static std::vector<std::vector<std::size_t>> read_plan(const std::filesystem::path& path){std::ifstream in(path);std::vector<std::vector<std::size_t>> result;std::string line;while(std::getline(in,line)){std::stringstream row(line);std::string field;for(int i=0;i<3;++i)std::getline(row,field,'\t');std::getline(row,field,'\t');std::stringstream blocks(field);std::vector<std::size_t> ids;while(std::getline(blocks,field,','))ids.push_back(std::stoul(field));result.push_back(ids);}if(result.size()!=28)throw std::runtime_error("plan layers");return result;}
static bool contains(const std::vector<std::size_t>& values,std::size_t value){return std::find(values.begin(),values.end(),value)!=values.end();}
static void json_array(std::ostream& out,const std::vector<std::size_t>& values){out<<'[';for(std::size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';}

int main(int argc,char** argv){try{
  std::filesystem::path prompt_reps,decode_reps,queries,store,plan,output="artifacts/cpp-p1-3c5-metrics.json";
  for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&](){if(++i>=argc)throw std::runtime_error("missing value");return std::filesystem::path(argv[i]);};if(a=="--prompt-representatives")prompt_reps=next();else if(a=="--decode-representatives")decode_reps=next();else if(a=="--queries")queries=next();else if(a=="--store")store=next();else if(a=="--plan")plan=next();else if(a=="--output")output=next();else throw std::runtime_error("argument: "+a);}
  constexpr std::size_t L=28,QH=16,D=128,B=17,BYTES=128*1024,BUDGET=4;
  auto prompt=read_all<float>(prompt_reps,L*QH*16*D),decode=read_all<float>(decode_reps,L*QH*D),query=read_all<float>(queries,L*QH*32*D);auto predicted=read_plan(plan);
  void* buffer=nullptr;if(posix_memalign(&buffer,4096,BUDGET*BYTES))throw std::bad_alloc();solidattention::UringReader reader(store.string(),{buffer},BUDGET*BYTES);
  std::size_t selected_new=0,dynamic_top4=0,hits=0,misses=0,verified=0,correction_verified=0;double selection_ms=0,read_ms=0,correction_read_ms=0;std::vector<std::size_t> ranks,first_selected,first_misses;
  for(std::size_t layer=0;layer<L;++layer){
    std::vector<float> merged(QH*B*D);for(std::size_t head=0;head<QH;++head){std::memcpy(merged.data()+(head*B)*D,prompt.data()+(layer*QH*16+head*16)*D,16*D*sizeof(float));std::memcpy(merged.data()+(head*B+16)*D,decode.data()+(layer*QH+head)*D,D*sizeof(float));}
    std::vector<float> last(QH*D);for(std::size_t head=0;head<QH;++head)std::memcpy(last.data()+head*D,query.data()+((layer*QH+head)*32+31)*D,D*sizeof(float));
    auto begin=std::chrono::steady_clock::now();auto selection=solidattention::select_shared_blocks(last.data(),merged.data(),QH,B,D,BUDGET,1,1);selection_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    selected_new+=contains(selection.block_ids,16);std::vector<std::size_t> ranked(B);for(std::size_t b=0;b<B;++b)ranked[b]=b;std::stable_sort(ranked.begin(),ranked.end(),[&](auto a,auto b){return selection.mean_head_scores[a]>selection.mean_head_scores[b];});auto rank=std::find(ranked.begin(),ranked.end(),16)-ranked.begin()+1;ranks.push_back(rank);dynamic_top4+=rank<=4;
    std::size_t layer_hits=0;std::vector<std::size_t> layer_misses;for(auto block:selection.block_ids){if(contains(predicted[layer],block))++layer_hits;else layer_misses.push_back(block);}hits+=layer_hits;misses+=layer_misses.size();
    std::vector<std::uint64_t> offsets;for(auto block:selection.block_ids)offsets.push_back((layer*B+block)*BYTES);read_ms+=reader.read_blocks_fixed(0,offsets,BYTES);
    std::ifstream source(store,std::ios::binary);std::vector<char> expected(BYTES);for(std::size_t slot=0;slot<BUDGET;++slot){source.seekg(offsets[slot]);source.read(expected.data(),BYTES);if(!source||std::memcmp(static_cast<char*>(buffer)+slot*BYTES,expected.data(),BYTES))throw std::runtime_error("selected block verification");++verified;}
    if(!layer_misses.empty()){
      std::vector<std::uint64_t> correction_offsets;for(auto block:layer_misses)correction_offsets.push_back((layer*B+block)*BYTES);
      correction_read_ms+=reader.read_blocks_fixed(0,correction_offsets,BYTES);
      for(std::size_t slot=0;slot<layer_misses.size();++slot){source.clear();source.seekg(correction_offsets[slot]);source.read(expected.data(),BYTES);if(!source||std::memcmp(static_cast<char*>(buffer)+slot*BYTES,expected.data(),BYTES))throw std::runtime_error("correction block verification");++correction_verified;}
    }
    if(layer==0){first_selected=selection.block_ids;first_misses=layer_misses;}
  }
  free(buffer);double mean_rank=0;for(auto rank:ranks)mean_rank+=rank;mean_rank/=ranks.size();std::ofstream out(output);out<<"{\n  \"version\": \"P1.3c.5-merged-selection-correction\",\n  \"layers\": 28,\n  \"candidate_blocks\": 17,\n  \"budget_blocks\": 4,\n  \"init_blocks\": 1,\n  \"local_blocks\": 1,\n  \"new_block_selected_layers\": "<<selected_new<<",\n  \"new_block_score_top4_layers\": "<<dynamic_top4<<",\n  \"new_block_mean_score_rank\": "<<mean_rank<<",\n  \"prediction_hits\": "<<hits<<",\n  \"correction_misses\": "<<misses<<",\n  \"correction_blocks_verified\": "<<correction_verified<<",\n  \"selected_blocks_verified\": "<<verified<<",\n  \"selection_ms_total\": "<<selection_ms<<",\n  \"selected_read_ms_total\": "<<read_ms<<",\n  \"correction_read_ms_total\": "<<correction_read_ms<<",\n  \"layer0_selected\": ";json_array(out,first_selected);out<<",\n  \"layer0_correction_misses\": ";json_array(out,first_misses);out<<"\n}\n";std::cout<<"selected_new="<<selected_new<<" score_top4="<<dynamic_top4<<" hits="<<hits<<" misses="<<misses<<" verified="<<verified<<" correction_verified="<<correction_verified<<"\n";
}catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
