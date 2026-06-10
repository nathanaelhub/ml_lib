# ------------------------------------------------------------------ #
#  ml_lib - tiny C machine learning library
# ------------------------------------------------------------------ #

CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -O2
LDFLAGS := -lm

TARGET  := ml_demo
OBJS    := ml_lib.o main.o
HEADERS := ml_lib.h

.PHONY: all clean run memcheck asan

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Each object depends on the header so edits to the API force a rebuild.
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

# Memory-leak verification using valgrind (Linux/macOS).
memcheck: $(TARGET)
	valgrind --leak-check=full --error-exitcode=1 ./$(TARGET)

# Alternative leak/UB check using compiler sanitizers (no valgrind needed).
asan: CFLAGS += -g -fsanitize=address,undefined -O1
asan: clean $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
