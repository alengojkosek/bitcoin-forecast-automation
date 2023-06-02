FROM  node:14-alpine
COPY src/client .
RUN npm install
EXPOSE 3000
CMD ["npm", "start"]