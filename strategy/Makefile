CPP=g++
BOOST_STAGE_LIB=/Users/gabrielsynnaeve/labs/boost_1_45_0/stage/lib
BOOST_INCLUDE=/Users/gabrielsynnaeve/labs/boost_1_45_0

tech_estimator: tech_estimator.cpp tech_estimator.h serialized_tables.h
	$(CPP) -arch i386 -I$(BOOST_INCLUDE) -L$(BOOST_STAGE_LIB) -lboost_serialization -o $@ $^ $(CFLAGS)

dumper: tech_estimator.cpp tech_estimator.h serialized_tables.h
	$(CPP) -arch i386 -DDUMP_ENUM -DDUMP_VECTOR_X -I$(BOOST_INCLUDE) -L$(BOOST_STAGE_LIB) -lboost_serialization -o $@ $^ $(CFLAGS)

dump: dumper
	@DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH):/Users/gabrielsynnaeve/labs/boost_1_45_0/stage/lib
	@name="PvP" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="PvT" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="PvZ" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="TvP" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="TvT" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="TvZ" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="ZvP" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="ZvT" && echo "dumping $$name" && ./dumper $$name > $$name.txt
	@name="ZvZ" && echo "dumping $$name" && ./dumper $$name > $$name.txt

.PHONY: clean

clean:
	rm -f *.o *~ tech_estimator
