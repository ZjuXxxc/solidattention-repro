#include "solidattention/selection.hpp"
#include "solidattention/uring_reader.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

template<class T> std::vector<T> read_all(const std::filesystem::path& p, std::size_t n) {
  std::vector<T> v(n); std::ifstream f(p,std::ios::binary);
  if(!f||!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))) throw std::runtime_error("short input");
  return v;
}
int main(int argc,char**argv){try{
  std::filesystem::path tail,queries,out="artifacts/cpp-p1-3c4";
  for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--tail"&&i+1<argc)tail=argv[++i];else if(a=="--queries"&&i+1<argc)queries=argv[++i];else if(a=="--output"&&i+1<argc)out=argv[++i];else throw std::runtime_error("argument");}
  constexpr std::size_t L=28,T=32,QH=16,KH=8,D=128,TB=2*KH*D*2,BB=T*TB,PB=17;
  auto kv=read_all<std::uint16_t>(tail,L*T*2*KH*D); auto q=read_all<float>(queries,L*QH*T*D);
  std::filesystem::create_directories(out); auto store=out/"main-kv-store.bin";
  int fd=open(store.c_str(),O_CREAT|O_TRUNC|O_RDWR|O_DIRECT,0644); if(fd<0)throw std::runtime_error("open");
  if(posix_fallocate(fd,0,L*PB*BB))throw std::runtime_error("fallocate");
  void* aligned=nullptr; if(posix_memalign(&aligned,4096,BB))throw std::bad_alloc();
  std::vector<float> reps; std::vector<std::size_t> offsets; std::size_t selected=0;
  for(std::size_t l=0;l<L;++l){
    auto r=solidattention::build_local_causal_representatives(q.data()+l*QH*T*D,kv.data()+l*T*2*KH*D,T,QH,KH,D,T,T,4);
    reps.insert(reps.end(),r.values.begin(),r.values.end()); offsets.insert(offsets.end(),r.chosen_token_offsets.begin(),r.chosen_token_offsets.end());
    std::vector<float> last(QH*D);for(std::size_t h=0;h<QH;++h)std::copy_n(q.data()+((l*QH+h)*T+T-1)*D,D,last.data()+h*D);
    auto s=solidattention::select_shared_blocks(last.data(),r.values.data(),QH,1,D,1,0,0); selected+=s.block_ids.size()==1&&s.block_ids[0]==0;
    std::memcpy(aligned,kv.data()+l*T*2*KH*D,BB); off_t pos=(l*PB+16)*BB;
    if(pwrite(fd,aligned,BB,pos)!=(ssize_t)BB)throw std::runtime_error("write");
  }
  fsync(fd);close(fd);
  auto representatives_path=out/"representatives.f32";
  int representatives_fd=open(representatives_path.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0644);
  if(representatives_fd<0)throw std::runtime_error("open representatives");
  const auto representatives_bytes=reps.size()*sizeof(float);
  if(write(representatives_fd,reps.data(),representatives_bytes)!=(ssize_t)representatives_bytes)throw std::runtime_error("write representatives");
  if(fsync(representatives_fd))throw std::runtime_error("fsync representatives");
  close(representatives_fd);
  void* rb=nullptr;if(posix_memalign(&rb,4096,BB))throw std::bad_alloc(); auto reader=std::make_unique<solidattention::UringReader>(store.string(),std::vector<void*>{rb},BB);std::size_t verified=0;
  for(std::size_t l=0;l<L;++l){reader->read_fixed(0,(l*PB+16)*BB);if(std::memcmp(rb,kv.data()+l*T*2*KH*D,BB))throw std::runtime_error("readback");++verified;}
  reader.reset();free(rb);free(aligned);
  auto temporary=out/"generation.json.tmp";auto committed=out/"generation.json";
  const std::string marker_body="{\"generation\":1,\"block_id\":16,\"layers\":28}\n";
  int marker_fd=open(temporary.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0644);
  if(marker_fd<0)throw std::runtime_error("open commit marker");
  if(write(marker_fd,marker_body.data(),marker_body.size())!=(ssize_t)marker_body.size())throw std::runtime_error("write commit marker");
  if(fsync(marker_fd))throw std::runtime_error("fsync commit marker");
  close(marker_fd);
  std::filesystem::rename(temporary,committed);
  int directory_fd=open(out.c_str(),O_RDONLY|O_DIRECTORY);
  if(directory_fd<0)throw std::runtime_error("open output directory");
  if(fsync(directory_fd))throw std::runtime_error("fsync output directory");
  close(directory_fd);
  std::ofstream m(out/"metrics.json");m<<"{\n  \"version\": \"P1.3c.4-online-seal-selection\",\n  \"sealed_blocks\": 28,\n  \"verified_reads\": "<<verified<<",\n  \"selection_generation\": 1,\n  \"new_block_id\": 16,\n  \"new_block_selected_layers\": "<<selected<<",\n  \"representative_values\": "<<reps.size()<<",\n  \"representative_indices\": "<<offsets.size()<<",\n  \"crash_consistent_commit_marker\": true\n}\n";
  std::cout<<"sealed=28 selected="<<selected<<" verified="<<verified<<"\n";
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
