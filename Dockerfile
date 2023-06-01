FROM node:14-alpine

WORKDIR /app

COPY src/client/package.json src/client/package-lock.json /app/
RUN npm install

COPY src/client /app/src/client

EXPOSE 3000

CMD ["sh", "-c", "cd src/client && npm start"]
