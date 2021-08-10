IMG_NAME = nam012-th19
BUILD_CMD = docker build --network=host
DOCKERFILE = Dockerfile

USERNAME = $(shell whoami)
USERID = $(shell id -u)
GROUPID = $(shell id -g)

BUILD_ARGS = --build-arg user=$(USERNAME) \
             --build-arg uid=$(USERID) \
             --build-arg gid=$(GROUPID)

docker:
	$(BUILD_CMD) $(BUILD_ARGS) -f $(DOCKERFILE) -t $(IMG_NAME) .

clean:
	docker image rm $(IMG_NAME)


